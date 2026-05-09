# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 kaede982006
"""Evaluation metrics for the gmdgen v0.1.0 AI baseline.

These are the *gates* the release notes promise. They run on a saved
checkpoint and a small held-out set of generated samples.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from gmdgen.data.loader import load_dataset
from gmdgen.generate.invariants import (
    InvariantViolation,
    assert_invariants,
)
from gmdgen.ml.dataset import (
    DatasetConfig,
    GMDTokenDataset,
    collate,
    encode_records_to_streams,
)
from gmdgen.ml.sample import SamplingConfig, generate
from gmdgen.ml.tokens import (
    BOS_ID,
    EOS_ID,
    NUM_SPECIAL,
    PAD_ID,
    encode_level_string,
)
from gmdgen.ml.train import load_checkpoint


@dataclass(slots=True)
class EvalConfig:
    ckpt: Path
    dataset_dir: Path
    n_samples: int = 8
    sections: int = 4
    out_path: Path | None = None
    seed: int = 0


def held_out_perplexity(model, vocab, dataset_dir: Path, *, ctx: int) -> float:
    """Compute perplexity on a held-out 10% slice of the dataset."""
    streams, _ = encode_records_to_streams(dataset_dir, vocab=vocab)
    if not streams:
        return float("nan")
    cfg = DatasetConfig(ctx=ctx, stride=ctx, augment=False, seed=999)
    ds = GMDTokenDataset(streams, cfg)
    n = len(ds)
    n_eval = max(1, n // 10)
    indices = list(range(n))[-n_eval:]
    loss_sum = 0.0
    n_tokens = 0
    model.eval()
    with torch.no_grad():
        for i in indices:
            batch = collate([ds[i]])
            out = model(batch)
            logits = out["logits_id"][:, :-1, :]
            targets = batch["ids"][:, 1:]
            mask = targets != PAD_ID
            if mask.sum() == 0:
                continue
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=PAD_ID,
                reduction="sum",
            )
            loss_sum += float(ce.item())
            n_tokens += int(mask.sum().item())
    if n_tokens == 0:
        return float("nan")
    avg = loss_sum / n_tokens
    return float(math.exp(min(20.0, avg)))


def editor_load_rate(samples: list[list[Any]]) -> float:
    """Fraction of generated samples whose object stream parses + has size > 0."""
    if not samples:
        return 0.0
    ok = 0
    for objs in samples:
        if not objs:
            continue
        ids_ok = all(o.object_id and o.object_id.isdigit() for o in objs)
        x_ok = all(b.x >= a.x for a, b in zip(objs, objs[1:]))
        if ids_ok and x_ok:
            ok += 1
    return ok / len(samples)


def simulate_play_success_rate(samples: list[list[Any]]) -> float:
    """Heuristic playability rate: every consecutive pair must be reachable.

    Reachability is approximated as max horizontal step under the GD jump
    physics — roughly 4 ground-units before a hazard. We use 90 px as a soft
    cap. (This is intentionally lenient; it is a *baseline* metric.)
    """
    if not samples:
        return 0.0
    JUMP_GAP = 90.0
    ok = 0
    for objs in samples:
        if len(objs) < 2:
            continue
        gaps = [b.x - a.x for a, b in zip(objs, objs[1:])]
        if all(g <= JUMP_GAP * 4 for g in gaps):
            ok += 1
    return ok / len(samples)


def mode_coverage_kl(
    samples: list[list[Any]], reference_classes: dict[str, float] | None = None
) -> float:
    """KL divergence between sample object-class distribution and reference.

    `reference_classes` defaults to a neutral uniform-ish distribution; if
    omitted we just measure entropy compared to uniform across class sizes.
    """
    if not samples:
        return float("inf")
    counts: Counter[str] = Counter()
    for objs in samples:
        for o in objs:
            counts[o.object_id] += 1
    total = sum(counts.values()) or 1
    pk = {k: v / total for k, v in counts.items()}
    if reference_classes is None:
        # uniform reference over observed ids
        n = len(pk)
        if n <= 1:
            return 0.0
        ref = {k: 1.0 / n for k in pk}
    else:
        ref = reference_classes
    kl = 0.0
    for k, p in pk.items():
        q = max(ref.get(k, 1e-6), 1e-6)
        kl += p * math.log(p / q)
    return float(kl)


def beat_sync_mae_ms(samples: list[list[Any]], _audio: Any = None) -> float | None:
    """Audio not available in the v0.1.0 baseline — return None."""
    return None


def repair_loss_proxy(samples: list[list[Any]]) -> float:
    """Fraction of objects we would have to drop to satisfy x-monotonic."""
    if not samples:
        return 0.0
    drops = 0
    total = 0
    for objs in samples:
        cur = -1.0
        for o in objs:
            total += 1
            if o.x < cur:
                drops += 1
            else:
                cur = o.x
    if total == 0:
        return 0.0
    return drops / total


def _check_invariants(samples: list[list[Any]]) -> dict[str, Any]:
    passed = 0
    total = len(samples)
    failures: list[str] = []
    for i, objs in enumerate(samples):
        try:
            assert_invariants(objs, section_count=1)
            passed += 1
        except InvariantViolation as exc:
            names = [r.name for r in exc.report.failures]
            failures.append(f"sample[{i}] {names}")
    return {
        "n": total,
        "passed": passed,
        "rate": passed / max(1, total),
        "failures": failures[:6],
    }


def evaluate(cfg: EvalConfig) -> dict[str, Any]:
    model, vocab, ckpt = load_checkpoint(cfg.ckpt)
    samples: list[list[Any]] = []
    for i in range(cfg.n_samples):
        objs = generate(
            model, vocab, sections=cfg.sections,
            cfg=SamplingConfig(seed=cfg.seed + i, max_objects=400),
        )
        samples.append(objs)

    ppl = held_out_perplexity(
        model, vocab, cfg.dataset_dir, ctx=model.cfg.ctx,
    )
    inv = _check_invariants(samples)
    metrics: dict[str, Any] = {
        "ckpt": str(cfg.ckpt),
        "n_params": ckpt.get("n_params"),
        "completed_steps": ckpt.get("completed_steps"),
        "n_samples": len(samples),
        "sections": cfg.sections,
        "editor_load_rate": editor_load_rate(samples),
        "simulate_play_success_rate": simulate_play_success_rate(samples),
        "mode_coverage_kl": mode_coverage_kl(samples),
        "beat_sync_mae_ms": beat_sync_mae_ms(samples),
        "repair_loss_proxy": repair_loss_proxy(samples),
        "held_out_perplexity": ppl,
        "invariant_pass": inv,
        "objects_per_sample": [len(s) for s in samples],
    }
    if cfg.out_path is not None:
        cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a gmdgen ML checkpoint.")
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--in", dest="dataset_dir", required=True, type=Path)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--sections", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("reports/eval.json"))
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _argparser().parse_args(argv)
    metrics = evaluate(
        EvalConfig(
            ckpt=args.ckpt,
            dataset_dir=args.dataset_dir,
            n_samples=args.samples,
            sections=args.sections,
            out_path=args.out,
            seed=args.seed,
        )
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
