# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 kaede982006
"""Smoke test entry point: ``python -m gmdgen.ml.smoke``.

This is the convergence check the recursive debug loop targets. It must be
runnable standalone (no dataset, no checkpoint), and only verifies that:

  1. the model can be constructed,
  2. one forward + backward + optimizer step succeeds,
  3. one round of sampling with a tiny IdVocab returns >= 1 object.

If anything fails, exit code is non-zero and the traceback is printed.
"""
from __future__ import annotations

import sys
import traceback

import torch

from gmdgen.ml.architectures import GMDLanguageModel, ModelConfig
from gmdgen.ml.sample import SamplingConfig, generate
from gmdgen.ml.tokens import BOS_ID, EOS_ID, IdVocab, NUM_SPECIAL
from gmdgen.utils.device import get_best_device, get_device_info


def _build_dummy_batch(cfg: ModelConfig, T: int = 16, B: int = 2) -> dict[str, torch.Tensor]:
    device = get_best_device()
    g = torch.Generator(device=device).manual_seed(0)
    return {
        "ids": torch.randint(NUM_SPECIAL, cfg.id_vocab, (B, T), generator=g, device=device),
        "cls": torch.randint(0, cfg.cls_vocab, (B, T), generator=g, device=device),
        "dx": torch.randint(0, cfg.dx_vocab, (B, T), generator=g, device=device),
        "y": torch.randint(0, cfg.y_vocab, (B, T), generator=g, device=device),
        "mode": torch.randint(0, cfg.mode_vocab, (B, T), generator=g, device=device),
        "speed": torch.randint(0, cfg.speed_vocab, (B, T), generator=g, device=device),
        "section": torch.randint(0, cfg.section_vocab, (B, T), generator=g, device=device),
    }


def _build_tiny_vocab() -> IdVocab:
    return IdVocab(id_to_slot={
        "1": NUM_SPECIAL + 0,
        "2": NUM_SPECIAL + 1,
        "12": NUM_SPECIAL + 2,
        "13": NUM_SPECIAL + 3,
        "201": NUM_SPECIAL + 4,
        "202": NUM_SPECIAL + 5,
    })


def main() -> int:
    try:
        torch.manual_seed(0)
        device = get_best_device()
        device_info = get_device_info()
        print(f"[smoke] device={device_info.compute_device} gpu_available={device_info.gpu_available}")
        
        cfg = ModelConfig(d_model=64, n_layer=2, n_head=2, d_ff=128, ctx=64)
        model = GMDLanguageModel(cfg)
        model.to(device)
        n = model.num_parameters()
        print(f"[smoke] params={n:,}")
        assert n > 0, "model has no parameters"

        # Forward
        batch = _build_dummy_batch(cfg)
        out = model(batch)
        for key in ("logits_id", "logits_dx", "logits_y", "h"):
            assert key in out, f"missing output key: {key}"
        print(f"[smoke] forward shapes "
              f"id={tuple(out['logits_id'].shape)} dx={tuple(out['logits_dx'].shape)} "
              f"y={tuple(out['logits_y'].shape)} h={tuple(out['h'].shape)}")

        # Backward + optim step
        loss = out["logits_id"].mean() + out["logits_dx"].mean() + out["logits_y"].mean()
        loss.backward()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        opt.step()
        opt.zero_grad()
        print(f"[smoke] backward+step ok, loss={float(loss):.4f}")

        # Sample
        vocab = _build_tiny_vocab()
        objects = generate(model, vocab, sections=2, cfg=SamplingConfig(max_objects=32, seed=0))
        print(f"[smoke] generated {len(objects)} objects")
        assert objects, "generate produced nothing"
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
