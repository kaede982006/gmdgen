# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from pathlib import Path

from gmdgen.feedback.store import (
    FeedbackRecord,
    export_feedback_eval_dataset,
    load_feedback_records,
    low_rated_feedback_context,
    save_feedback_record,
)


def test_feedback_record_saved_without_api_key(tmp_path: Path) -> None:
    path = tmp_path / "feedback.jsonl"
    save_feedback_record(
        FeedbackRecord(
            input_summary={"ollama_base_url": "sk-secret", "audio_file": "C:/Users/xisik/song.wav"},
            user_rating=5,
            user_notes="good",
        ),
        path,
    )

    text = path.read_text(encoding="utf-8")
    assert "sk-secret" not in text
    assert "ollama_base_url" not in text


def test_feedback_tags_saved(tmp_path: Path) -> None:
    path = tmp_path / "feedback.jsonl"
    save_feedback_record(FeedbackRecord(user_rating=2, user_tags=["too_empty", "off_sync"]), path)

    records = load_feedback_records(path)

    assert records[0]["user_tags"] == ["too_empty", "off_sync"]


def test_feedback_can_export_eval_dataset(tmp_path: Path) -> None:
    feedback = tmp_path / "feedback.jsonl"
    output = tmp_path / "eval.json"
    save_feedback_record(FeedbackRecord(user_rating=5, audio_summary={"bpm": 120}, final_plan={"object_plans": []}), feedback)

    export_feedback_eval_dataset(feedback, output, min_rating=4)

    assert output.exists()
    assert "audio_summary" in output.read_text(encoding="utf-8")


def test_low_rated_outputs_feed_future_prompt_context(tmp_path: Path) -> None:
    path = tmp_path / "feedback.jsonl"
    save_feedback_record(FeedbackRecord(user_rating=1, user_tags=["bad_drop"], user_notes="drop was empty"), path)

    context = low_rated_feedback_context(path)

    assert "bad_drop" in context[0]
    assert "drop was empty" in context[0]
