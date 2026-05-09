# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 kaede982006
"""Training loop for the GMD language model.

Implements next-object cross-entropy with auxiliary dx/y prediction
(Foundations of LLMs §1.1.1 — self-supervised, §1.2.1 — decoder-only LM).
Optimizer is AdamW with a cosine schedule and warmup (Fleuret §3.3).

This is intentionally simple: a single training process, in-memory dataset,
no DDP, no mixed precision. The dataset is small (~129 .gmd) so a 1-3M
parameter model converges within a few thousand steps on CPU.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from gmdgen.ml.architectures import GMDLanguageModel, ModelConfig
from gmdgen.ml.dataset import (
    DatasetConfig,
    GMDTokenDataset,
    collate,
    encode_records_to_streams,
)
from gmdgen.ml.tokens import IdVocab, PAD_ID


@dataclass(slots=True)
class TrainConfig:
    dataset_dir: Path
    out_ckpt: Path
    max_steps: int = 2000
    batch_size: int = 16
    ctx: int = 512
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    log_every: int = 50
    eval_every: int = 200
    val_split: float = 0.1
    seed: int = 0
    log_jsonl: Path | None = None
    aux_dx_weight: float = 0.3
    aux_y_weight: float = 0.3


def _lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _shifted_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logits: [B, T, V]; targets: [B, T]
    B, T, V = logits.shape
    # next-token: predict targets[:, 1:] from logits[:, :-1]
    pred = logits[:, :-1, :].reshape(-1, V)
    gt = targets[:, 1:].reshape(-1)
    return F.cross_entropy(pred, gt, ignore_index=PAD_ID)


def _step_loss(
    model: GMDLanguageModel,
    batch: dict[str, torch.Tensor],
    cfg: TrainConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    out = model(batch)
    loss_id = _shifted_ce(out["logits_id"], batch["ids"])
    loss_dx = _shifted_ce(out["logits_dx"], batch["dx"])
    loss_y = _shifted_ce(out["logits_y"], batch["y"])
    loss = loss_id + cfg.aux_dx_weight * loss_dx + cfg.aux_y_weight * loss_y
    return loss, {
        "loss_id": float(loss_id.detach().item()),
        "loss_dx": float(loss_dx.detach().item()),
        "loss_y": float(loss_y.detach().item()),
        "loss": float(loss.detach().item()),
    }


def _evaluate(
    model: GMDLanguageModel,
    loader: DataLoader,
    cfg: TrainConfig,
    max_batches: int = 8,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            _, stats = _step_loss(model, batch, cfg)
            losses.append(stats["loss_id"])
    model.train()
    avg = sum(losses) / max(1, len(losses))
    return {"val_loss_id": avg, "val_perplexity": float(math.exp(min(20.0, avg)))}


def train(cfg: TrainConfig) -> dict[str, Any]:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(1)

    streams, vocab = encode_records_to_streams(cfg.dataset_dir)
    if not streams:
        raise RuntimeError("No GMD streams produced from the dataset.")

    ds_cfg = DatasetConfig(ctx=cfg.ctx, stride=max(64, cfg.ctx // 2), seed=cfg.seed)
    full_ds = GMDTokenDataset(streams, ds_cfg)

    n = len(full_ds)
    n_val = max(1, int(n * cfg.val_split))
    n_train = max(1, n - n_val)
    g = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate,
        drop_last=False,
    )

    model_cfg = ModelConfig(ctx=cfg.ctx)
    model = GMDLanguageModel(model_cfg)
    n_params = model.num_parameters()

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
    )

    log_path = cfg.log_jsonl
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = log_path.open("a", encoding="utf-8")
    else:
        log_fp = None

    def _log(record: dict[str, Any]) -> None:
        record = {"ts": time.time(), **record}
        if log_fp:
            log_fp.write(json.dumps(record) + "\n")
            log_fp.flush()

    _log({"event": "start", "n_params": n_params, "n_train": n_train, "n_val": n_val})

    model.train()
    step = 0
    last_eval: dict[str, float] = {}
    try:
        while step < cfg.max_steps:
            for batch in train_loader:
                if step >= cfg.max_steps:
                    break
                lr = _lr_at(step, cfg)
                for g in opt.param_groups:
                    g["lr"] = lr

                loss, stats = _step_loss(model, batch, cfg)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                if step % cfg.log_every == 0:
                    msg = (
                        f"step={step:5d}  lr={lr:.2e}  "
                        f"loss={stats['loss']:.3f}  id={stats['loss_id']:.3f}  "
                        f"dx={stats['loss_dx']:.3f}  y={stats['loss_y']:.3f}"
                    )
                    print(msg, flush=True)
                    _log({"event": "step", "step": step, "lr": lr, **stats})

                if step > 0 and step % cfg.eval_every == 0:
                    last_eval = _evaluate(model, val_loader, cfg)
                    print(
                        f"  [eval] val_loss_id={last_eval['val_loss_id']:.3f}  "
                        f"ppl={last_eval['val_perplexity']:.2f}",
                        flush=True,
                    )
                    _log({"event": "eval", "step": step, **last_eval})

                step += 1
    finally:
        last_eval = _evaluate(model, val_loader, cfg) or last_eval
        cfg.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "state_dict": model.state_dict(),
            "model_cfg": asdict(model_cfg),
            "vocab": vocab.to_dict(),
            "train_cfg": {
                "max_steps": cfg.max_steps,
                "batch_size": cfg.batch_size,
                "ctx": cfg.ctx,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "seed": cfg.seed,
            },
            "n_params": n_params,
            "final_eval": last_eval,
            "completed_steps": step,
        }
        torch.save(ckpt, cfg.out_ckpt)
        _log({"event": "saved", "ckpt": str(cfg.out_ckpt), **last_eval})
        if log_fp:
            log_fp.close()

    return {
        "ckpt": str(cfg.out_ckpt),
        "n_params": n_params,
        "completed_steps": step,
        "final_eval": last_eval,
    }


def load_checkpoint(path: Path) -> tuple[GMDLanguageModel, IdVocab, dict[str, Any]]:
    """Load a checkpoint for inference."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    model = GMDLanguageModel(model_cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    vocab = IdVocab.from_dict(ckpt["vocab"])
    return model, vocab, ckpt


def _argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the gmdgen language model.")
    p.add_argument("--in", dest="in_dir", required=True, type=Path)
    p.add_argument("--out", dest="out_ckpt", required=True, type=Path)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--batch", dest="batch_size", type=int, default=16)
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-jsonl", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _argparser().parse_args(argv)
    cfg = TrainConfig(
        dataset_dir=args.in_dir,
        out_ckpt=args.out_ckpt,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        ctx=args.ctx,
        lr=args.lr,
        seed=args.seed,
        log_jsonl=args.log_jsonl,
    )
    res = train(cfg)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
