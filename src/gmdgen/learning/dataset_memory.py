# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from pathlib import Path
from typing import Any

from gmdgen.ai.dataset_index import build_dataset_index, ensure_dataset_dir
from gmdgen.learning.feature_extractor import (
    export_finetune_jsonl_from_learning_store,
    export_preference_pairs_from_learning_store,
)
from gmdgen.learning.store import (
    load_learning_examples,
    save_learning_example,
    update_learning_example_feedback,
)


def dataset_learning_examples_dir(dataset_dir: str | Path | None = None) -> Path:
    return ensure_dataset_dir(dataset_dir) / "learning" / "examples"


def dataset_learning_feedback_dir(dataset_dir: str | Path | None = None) -> Path:
    return ensure_dataset_dir(dataset_dir) / "learning" / "feedback"


def dataset_learning_exports_dir(dataset_dir: str | Path | None = None) -> Path:
    return ensure_dataset_dir(dataset_dir) / "learning" / "exports"


def dataset_learned_data_store_dir(dataset_dir: str | Path | None = None) -> Path:
    return ensure_dataset_dir(dataset_dir) / "learning" / "style_memory"


def save_learning_example_to_dataset(example: dict[str, Any] | Any, dataset_dir: str | Path | None = None) -> str:
    return save_learning_example(example, store_dir=dataset_learning_examples_dir(dataset_dir))


def update_dataset_feedback(example_id: str, feedback: dict[str, Any], dataset_dir: str | Path | None = None) -> bool:
    feedback_dir = dataset_learning_feedback_dir(dataset_dir)
    feedback_dir.mkdir(parents=True, exist_ok=True)
    return update_learning_example_feedback(
        example_id,
        feedback,
        store_dir=dataset_learning_examples_dir(dataset_dir),
    )


def load_dataset_learning_examples(
    dataset_dir: str | Path | None = None,
    *,
    limit: int | None = None,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return load_learning_examples(
        store_dir=dataset_learning_examples_dir(dataset_dir),
        limit=limit,
        filters=filters,
    )


def rebuild_index_after_learning_save(dataset_dir: str | Path | None = None) -> dict[str, Any]:
    index = build_dataset_index(ensure_dataset_dir(dataset_dir))
    return index.to_dict()


def export_finetune_jsonl_from_dataset(dataset_dir: str | Path | None, output_path: str | Path) -> Path:
    return export_finetune_jsonl_from_learning_store(
        output_path,
        store_dir=dataset_learned_data_store_dir(dataset_dir),
    )


def export_preference_pairs_from_dataset(dataset_dir: str | Path | None, output_path: str | Path) -> Path:
    return export_preference_pairs_from_learning_store(
        output_path,
        store_dir=dataset_learned_data_store_dir(dataset_dir),
    )
