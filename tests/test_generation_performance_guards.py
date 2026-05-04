# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

"""Tests for generation performance guard configuration and bounds."""

from dataclasses import fields, is_dataclass
from gmdgen.gui.app import GuiGenerationConfig


def _field_names() -> set[str]:
    if is_dataclass(GuiGenerationConfig):
        return {f.name for f in fields(GuiGenerationConfig)}
    import inspect
    sig = inspect.signature(GuiGenerationConfig.__init__)
    return {p for p in sig.parameters if p != "self"}


def test_config_has_max_extreme_ml_seconds():
    assert "max_extreme_ml_seconds" in _field_names()


def test_config_has_ai_timeout_seconds():
    assert "ai_timeout_seconds" in _field_names()


def test_config_has_ai_candidate_count():
    assert "ai_candidate_count" in _field_names()


def test_config_has_ai_max_regeneration_attempts():
    assert "ai_max_regeneration_attempts" in _field_names()


def test_max_extreme_ml_seconds_default_is_sane():
    cfg = GuiGenerationConfig(audio_file="x.mp3", output_path="out.gmd")
    assert 60 <= cfg.max_extreme_ml_seconds <= 3600, (
        f"max_extreme_ml_seconds={cfg.max_extreme_ml_seconds} is outside [60, 3600]"
    )


def test_ai_timeout_default_is_sane():
    cfg = GuiGenerationConfig(audio_file="x.mp3", output_path="out.gmd")
    assert 10 <= cfg.ai_timeout_seconds <= 600, (
        f"ai_timeout_seconds={cfg.ai_timeout_seconds} is outside [10, 600]"
    )


def test_ai_candidate_count_default_is_bounded():
    cfg = GuiGenerationConfig(audio_file="x.mp3", output_path="out.gmd")
    assert 1 <= cfg.ai_candidate_count <= 20, (
        f"ai_candidate_count={cfg.ai_candidate_count} is outside [1, 20]"
    )


def test_ai_retry_count_default_bounded():
    cfg = GuiGenerationConfig(audio_file="x.mp3", output_path="out.gmd")
    assert 0 <= cfg.ai_retry_count <= 10, (
        f"ai_retry_count={cfg.ai_retry_count} is outside [0, 10]"
    )


def test_max_output_tokens_default_bounded():
    cfg = GuiGenerationConfig(audio_file="x.mp3", output_path="out.gmd")
    assert 256 <= cfg.max_output_tokens <= 32768, (
        f"max_output_tokens={cfg.max_output_tokens} is outside [256, 32768]"
    )


def test_object_budget_default_is_sane():
    cfg = GuiGenerationConfig(audio_file="x.mp3", output_path="out.gmd")
    assert 10 <= cfg.object_budget <= 100000, (
        f"object_budget={cfg.object_budget} is outside safe range"
    )


def test_config_to_generation_dict_includes_timing_keys():
    """The dict produced by to_generation_config must include timing bounds."""
    cfg = GuiGenerationConfig(audio_file="x.mp3", output_path="out.gmd")
    gen_config = cfg.to_generation_config()
    assert "max_extreme_ml_seconds" in gen_config, (
        "Generation config dict must include max_extreme_ml_seconds"
    )


def test_beat_snap_tolerance_is_positive():
    cfg = GuiGenerationConfig(audio_file="x.mp3", output_path="out.gmd")
    assert cfg.beat_snap_tolerance > 0.0


def test_sync_strength_in_valid_range():
    cfg = GuiGenerationConfig(audio_file="x.mp3", output_path="out.gmd")
    assert 0.0 <= cfg.sync_strength <= 1.0


def test_candidates_per_section_bounded():
    cfg = GuiGenerationConfig(audio_file="x.mp3", output_path="out.gmd")
    assert 1 <= cfg.candidates_per_section <= 20, (
        f"candidates_per_section={cfg.candidates_per_section} out of expected range"
    )
