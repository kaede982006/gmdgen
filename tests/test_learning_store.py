# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
from pathlib import Path

from gmdgen.learning.store import (
    LearningExample,
    build_learning_example_from_result,
    learning_store_path,
    load_learning_examples,
    save_learning_example,
    select_failure_examples_for_prompt_feedback,
    select_high_quality_examples,
    summarize_learning_examples_for_context,
    update_learning_example_feedback,
)
from gmdgen.generate.audio_conditioned import _append_learning_context


def test_learning_example_saved_without_api_key(tmp_path: Path) -> None:
    example_id = save_learning_example(
        LearningExample(
            generation_config_summary={
                "ollama_base_url": "sk-secret",
                "audio_file": "C:/Users/xisik/Music/song.wav",
            },
            user_rating=5,
        ),
        store_dir=tmp_path,
    )

    text = learning_store_path(tmp_path).read_text(encoding="utf-8")

    assert example_id
    assert "sk-secret" not in text
    assert "ollama_base_url" not in text


def test_learning_example_does_not_store_absolute_paths_by_default(tmp_path: Path) -> None:
    save_learning_example(
        {
            "example_id": "path-test",
            "generation_config_summary": {
                "audio_file": "C:/Users/xisik/Music/my song.wav",
                "output_path": "C:/Users/xisik/Projects/xmaker/outputs/out.gmd",
            },
        },
        store_dir=tmp_path,
    )

    payload = json.loads(learning_store_path(tmp_path).read_text(encoding="utf-8").splitlines()[0])

    assert payload["generation_config_summary"]["audio_file"] == "my song.wav"
    assert payload["generation_config_summary"]["output_path"] == "out.gmd"


def test_learning_store_loads_recent_examples(tmp_path: Path) -> None:
    for index in range(4):
        save_learning_example({"example_id": f"ex-{index}", "user_rating": index}, store_dir=tmp_path)

    recent = load_learning_examples(store_dir=tmp_path, limit=2)

    assert [item["example_id"] for item in recent] == ["ex-2", "ex-3"]


def test_high_quality_examples_selected_for_context(tmp_path: Path) -> None:
    examples = [
        {"example_id": "bad", "user_rating": 1, "score_breakdown": {"total": 0.2}},
        {"example_id": "good", "user_rating": 5, "score_breakdown": {"total": 0.9}},
    ]

    selected = select_high_quality_examples(examples)

    assert selected[0]["example_id"] == "good"


def test_failure_examples_selected_for_feedback() -> None:
    examples = [
        {"example_id": "good", "user_rating": 5},
        {"example_id": "bad", "user_rating": 1, "user_tags": ["too_empty"]},
        {"example_id": "loss", "quality_loss_reasons": ["drop too weak"]},
    ]

    selected = select_failure_examples_for_prompt_feedback(examples)

    assert {item["example_id"] for item in selected} == {"bad", "loss"}


def test_learning_store_survives_corrupt_record(tmp_path: Path) -> None:
    path = learning_store_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not-json\n{\"example_id\": \"ok\"}\n", encoding="utf-8")

    records = load_learning_examples(store_dir=tmp_path)

    assert records == [{"example_id": "ok"}]


def test_feedback_updates_learning_example(tmp_path: Path) -> None:
    example_id = save_learning_example({"example_id": "feedback", "user_rating": 0}, store_dir=tmp_path)

    updated = update_learning_example_feedback(
        example_id,
        {
            "user_rating": 2,
            "user_tags": ["bad_drop"],
            "user_notes": "drop was weak",
            "accepted_for_training": False,
        },
        store_dir=tmp_path,
    )
    record = load_learning_examples(store_dir=tmp_path)[0]

    assert updated is True
    assert record["user_rating"] == 2
    assert record["user_tags"] == ["bad_drop"]
    assert record["accepted_for_training"] is False


def test_low_rating_used_as_failure_memory(tmp_path: Path) -> None:
    save_learning_example(
        {"example_id": "bad", "user_rating": 1, "user_tags": ["off_sync"], "user_notes": "missed beat"},
        store_dir=tmp_path,
    )

    context = summarize_learning_examples_for_context(store_dir=tmp_path)

    assert context.failure_patterns_summary
    assert "off_sync" in context.failure_patterns_summary[0]


def test_high_rating_used_as_success_memory(tmp_path: Path) -> None:
    save_learning_example(
        {
            "example_id": "good",
            "user_rating": 5,
            "score_breakdown": {"total": 0.91},
            "prompt_summary": "strong drop sync",
        },
        store_dir=tmp_path,
    )

    context = summarize_learning_examples_for_context(store_dir=tmp_path)

    assert context.good_examples_summary
    assert context.good_examples_summary[0]["rating"] == 5


def test_learning_context_respects_max_chars(tmp_path: Path) -> None:
    for index in range(10):
        save_learning_example(
            {
                "example_id": f"ex-{index}",
                "user_rating": 5,
                "prompt_summary": "x" * 500,
                "quality_loss_reasons": ["too_empty"] * 5,
            },
            store_dir=tmp_path,
        )

    context = summarize_learning_examples_for_context(store_dir=tmp_path, max_chars=400)

    assert len(json.dumps(context.to_dict(), ensure_ascii=False)) < 2500


def test_build_learning_example_from_result_sanitizes_config() -> None:
    example = build_learning_example_from_result(
        {
            "output_path": "C:/Users/xisik/Projects/xmaker/outputs/song.gmd",
            "score_breakdown": {"total": 0.8},
            "quality_loss_reason_summary": ["none"],
        },
        {"ollama_base_url": "sk-secret", "audio_file": "C:/Users/xisik/Music/song.wav"},
    )

    payload = example.to_dict()

    assert "ollama_base_url" not in payload["generation_config_summary"]
    assert payload["generation_config_summary"]["audio_file"] == "song.wav"
    assert payload["output_file_name"] == "song.gmd"


def test_learning_memory_context_included_in_prompt_chunks(tmp_path: Path) -> None:
    save_learning_example(
        {
            "example_id": "good",
            "user_rating": 5,
            "prompt_summary": "sync the drop with pulse accents",
            "score_breakdown": {"total": 0.9},
        },
        store_dir=tmp_path,
    )

    chunks = _append_learning_context(
        [{"path": "base", "title": "Base", "text": "base context"}],
        {"use_learning_memory": True, "learning_store_dir": str(tmp_path), "ollama_max_context_chars": 2000},
    )

    assert any(chunk["path"] == "learning_memory" for chunk in chunks)


def test_learning_memory_does_not_enable_local_generation(tmp_path: Path) -> None:
    save_learning_example({"example_id": "good", "user_rating": 5}, store_dir=tmp_path)

    chunks = _append_learning_context(
        [],
        {"use_learning_memory": True, "learning_store_dir": str(tmp_path), "ollama_max_context_chars": 2000},
    )

    assert all("local_test_only" not in json.dumps(chunk) for chunk in chunks)
