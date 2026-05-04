from __future__ import annotations

from pathlib import Path

from gmdgen.ai.fine_tune_export import export_high_quality_fine_tuning_examples
from gmdgen.ai.preference_export import build_preference_pair, export_preference_pairs, validate_preference_jsonl


def test_finetune_export_filters_low_quality_outputs(tmp_path: Path) -> None:
    path = tmp_path / "ft.jsonl"
    count = export_high_quality_fine_tuning_examples(
        [
            {"input": {}, "output": {"sections": [], "object_plans": [], "trigger_plans": [], "validation_score": 0.9, "repair_loss_ratio": 0.1, "user_rating": 5}},
            {"input": {}, "output": {"sections": [], "object_plans": [], "trigger_plans": [], "validation_score": 0.2, "repair_loss_ratio": 0.8, "user_rating": 1}},
        ],
        path,
    )

    assert count == 1
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_preference_export_chosen_rejected_pair(tmp_path: Path) -> None:
    pair = build_preference_pair(
        input_summary={"audio": "summary"},
        chosen_candidate={"score": 0.9},
        rejected_candidate={"score": 0.2},
        reason="better drop impact",
    )
    path = tmp_path / "prefs.jsonl"

    export_preference_pairs([pair], path)

    assert validate_preference_jsonl(path) == []


def test_export_excludes_api_key_and_absolute_paths(tmp_path: Path) -> None:
    pair = build_preference_pair(
        input_summary={"ollama_base_url": "sk-secret", "audio_file": "C:/Users/xisik/song.wav"},
        chosen_candidate={"score": 0.9},
        rejected_candidate={"score": 0.2},
        reason="ok",
    )
    path = tmp_path / "safe.jsonl"

    export_preference_pairs([pair], path)
    text = path.read_text(encoding="utf-8")

    assert "sk-secret" not in text
    assert "ollama_base_url" not in text
    assert "C:/Users" not in text
