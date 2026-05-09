# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 kaede982006
"""Smoke + shape tests for the GMDLanguageModel."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gmdgen.ml.architectures import GMDLanguageModel, ModelConfig
from gmdgen.ml.sample import SamplingConfig, generate, generate_with_diagnostics
from gmdgen.ml.tokens import IdVocab, NUM_SPECIAL


@pytest.fixture(scope="module")
def tiny_model() -> GMDLanguageModel:
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=32, n_layer=2, n_head=2, d_ff=64, ctx=32)
    return GMDLanguageModel(cfg)


def _dummy_batch(cfg: ModelConfig, T: int = 8, B: int = 2):
    g = torch.Generator().manual_seed(1)
    return {
        "ids": torch.randint(NUM_SPECIAL, cfg.id_vocab, (B, T), generator=g),
        "cls": torch.randint(0, cfg.cls_vocab, (B, T), generator=g),
        "dx": torch.randint(0, cfg.dx_vocab, (B, T), generator=g),
        "y": torch.randint(0, cfg.y_vocab, (B, T), generator=g),
        "mode": torch.randint(0, cfg.mode_vocab, (B, T), generator=g),
        "speed": torch.randint(0, cfg.speed_vocab, (B, T), generator=g),
        "section": torch.randint(0, cfg.section_vocab, (B, T), generator=g),
    }


def test_forward_shapes(tiny_model: GMDLanguageModel) -> None:
    cfg = tiny_model.cfg
    batch = _dummy_batch(cfg)
    out = tiny_model(batch)
    B, T = batch["ids"].shape
    assert out["logits_id"].shape == (B, T, cfg.id_vocab)
    assert out["logits_dx"].shape == (B, T, cfg.dx_vocab)
    assert out["logits_y"].shape == (B, T, cfg.y_vocab)
    assert out["h"].shape == (B, T, cfg.d_model)


def test_backward_runs(tiny_model: GMDLanguageModel) -> None:
    batch = _dummy_batch(tiny_model.cfg)
    out = tiny_model(batch)
    loss = out["logits_id"].mean() + out["logits_dx"].mean() + out["logits_y"].mean()
    loss.backward()
    grads = [p.grad for p in tiny_model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_param_count_under_5M(tiny_model: GMDLanguageModel) -> None:
    assert tiny_model.num_parameters() < 5_000_000


def test_generate_returns_objects(tiny_model: GMDLanguageModel) -> None:
    vocab = IdVocab(id_to_slot={"1": NUM_SPECIAL, "2": NUM_SPECIAL + 1,
                                "12": NUM_SPECIAL + 2, "201": NUM_SPECIAL + 3})
    objs = generate(
        tiny_model, vocab, sections=2,
        cfg=SamplingConfig(max_objects=16, seed=0),
    )
    assert len(objs) >= 1
    assert all(o.x >= 0 for o in objs)
    # x must be non-decreasing (we enforce monotonic at sample time)
    xs = [o.x for o in objs]
    assert all(b >= a for a, b in zip(xs, xs[1:]))


def test_generate_uses_prompt_mode_and_reports_diagnostics(tiny_model: GMDLanguageModel) -> None:
    vocab = IdVocab(id_to_slot={"1": NUM_SPECIAL, "13": NUM_SPECIAL + 1})
    objs, diagnostics = generate_with_diagnostics(
        tiny_model,
        vocab,
        sections=1,
        cfg=SamplingConfig(max_objects=8, seed=2, prompt="ship challenge", candidates=1),
    )
    assert any(o.object_id == "13" and o.role == "portal" for o in objs)
    assert diagnostics["candidate_width"] == 1
    assert diagnostics["final_object_count"] == len(objs)
    assert "repair_added_ratio" in diagnostics


def test_full_size_model_under_param_budget() -> None:
    cfg = ModelConfig()  # default 4 layers, d_model=128
    model = GMDLanguageModel(cfg)
    n = model.num_parameters()
    # Expectation: well under 5M params for the chosen size.
    assert n < 5_000_000, f"too many parameters: {n}"
    assert n > 100_000, f"suspiciously small: {n}"
