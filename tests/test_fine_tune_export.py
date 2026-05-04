# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gmdgen.ai.fine_tune_export import (
    build_example_from_generation_run,
    export_fine_tuning_examples,
    validate_fine_tuning_jsonl,
)


def test_fine_tune_export_writes_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "examples.jsonl"
    example = {
        "input": {"audio_summary": {"bpm": 120}},
        "output": {"sections": [], "object_plans": [], "trigger_plans": []},
    }

    export_fine_tuning_examples([example], path)

    assert path.exists()
    assert validate_fine_tuning_jsonl(path) == []


def test_fine_tune_export_validates_required_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires input and output"):
        export_fine_tuning_examples([{"input": {}}], tmp_path / "bad.jsonl")


def test_fine_tune_export_does_not_include_api_key(tmp_path: Path) -> None:
    path = tmp_path / "examples.jsonl"
    example = {
        "input": {"ollama_base_url": "sk-secret"},
        "output": {"sections": [], "object_plans": [], "trigger_plans": [], "note": "sk-secret"},
    }

    export_fine_tuning_examples([example], path)
    text = path.read_text(encoding="utf-8")

    assert "ollama_base_url" not in text
    assert "sk-secret" not in text


def test_fine_tune_export_uses_structured_output_not_raw_save_string(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="structured plans"):
        export_fine_tuning_examples(
            [{"input": {}, "output": {"raw_save_string": "1,1,2,0,3,0;"}}],
            tmp_path / "raw.jsonl",
        )


def test_build_example_from_generation_run_contains_audio_summary() -> None:
    result = {
        "audio_file_name": "song.wav",
        "audio_backend": "fallback_wav",
        "detected_bpm": 120,
        "beat_count": 4,
        "validation_report": {"score": 0.9},
        "section_plan": [],
        "time_x_report": {},
    }
    example = build_example_from_generation_run(result, config={"object_budget": 20})

    assert example["input"]["audio_summary"]["audio_file_name"] == "song.wav"
    json.dumps(example)
