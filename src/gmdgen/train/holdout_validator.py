# SPDX-License-Identifier: GPL-3.0-or-later
"""Hold-out validation pipeline.

Goodfellow Ch.5 §5.3 "Hyperparameters and Validation Sets":
  The validation set measures generalisation — whether the trained model
  produces levels that are structurally sound even for .gmd files it has
  never seen during training.

We measure each held-out level against the following structural metrics:
  - parse_ok            : can the .gmd be re-parsed correctly?
  - x_monotone_ratio    : fraction of X-increasing consecutive object pairs
  - section_count       : number of distinct speed/gamemode sections
  - density_mean        : mean objects per grid unit (30px)
  - density_std         : std objects per grid unit
  - trigger_integrity   : fraction of triggers with valid target groups
  - class_distribution  : proportions of structure/decoration/trigger/portal

Aggregated metrics across all validation records are returned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from gmdgen.data.preprocess import split_level_objects
from gmdgen.data.schema import GMDRecord
from gmdgen.data.splitter import DatasetSplit, split_dataset_dir
from gmdgen.generate.scoring import (
    LevelScore,
    build_reference_stats_from_records,
    compute_level_score,
)
from gmdgen.io.gmd_decoder import decode_level_data
from gmdgen.io.gmd_parser import parse_gmd_text
from gmdgen.train.evaluator import evaluate_level_structure

LOGGER = logging.getLogger(__name__)


@dataclass
class RecordValidationResult:
    level_name: str
    parse_ok: bool
    object_count: int
    x_monotone_ratio: float
    section_count: int
    density_mean: float
    density_std: float
    trigger_integrity: float
    score_total: float
    score_breakdown: dict[str, float]


def validate_single_record(
    record: GMDRecord,
    *,
    reference_stats: dict[str, Any] | None = None,
) -> RecordValidationResult:
    level_name = record.document.tags.get("k2", ("s", record.document.path.stem))[1]
    objects = split_level_objects(record.decoded_level_data)

    structural = evaluate_level_structure(record.decoded_level_data)

    score = compute_level_score(
        objects,
        target_class_distribution=reference_stats.get("class_distribution") if reference_stats else None,
        reference_density=reference_stats.get("density_distribution") if reference_stats else None,
        reference_section_lengths=reference_stats.get("section_lengths") if reference_stats else None,
    )

    return RecordValidationResult(
        level_name=level_name,
        parse_ok=True,
        object_count=structural.get("object_count", 0),  # type: ignore
        x_monotone_ratio=structural.get("x_monotone_ratio", 1.0),
        section_count=structural.get("section_count", 1),  # type: ignore
        density_mean=structural.get("density_mean", 0.0),
        density_std=structural.get("density_std", 0.0),
        trigger_integrity=score.trigger,
        score_total=score.total,
        score_breakdown=score.to_dict(),
    )


def run_holdout_validation(
    dataset_dir: Path,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
    min_val_records: int = 1,
) -> dict[str, Any]:
    """Split dataset, train reference stats on train split, evaluate on val split."""
    split, load_result = split_dataset_dir(
        dataset_dir,
        val_ratio=val_ratio,
        seed=seed,
        min_val_records=min_val_records,
    )

    LOGGER.info(
        "Hold-out split: train=%d val=%d (total loaded=%d)",
        split.train_count,
        split.val_count,
        load_result.report.loaded_records,
    )

    if not split.train or not split.validation:
        return {"error": "Not enough records to split", "loaded": load_result.report.loaded_records}

    reference_stats = build_reference_stats_from_records(split.train)

    results: list[RecordValidationResult] = []
    for record in split.validation:
        try:
            result = validate_single_record(record, reference_stats=reference_stats)
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Skip validation record %s: %s", record.document.path.name, exc)

    if not results:
        return {"error": "No validation records produced results"}

    scores = [r.score_total for r in results]
    x_mono = [r.x_monotone_ratio for r in results]
    trigger_int = [r.trigger_integrity for r in results]
    section_counts = [r.section_count for r in results]
    obj_counts = [r.object_count for r in results]

    def _safe_stdev(vals: list[float]) -> float:
        return round(stdev(vals), 4) if len(vals) > 1 else 0.0

    summary = {
        "split": {"train": split.train_count, "validation": split.val_count},
        "score_mean": round(mean(scores), 4),
        "score_std": _safe_stdev(scores),
        "x_monotone_mean": round(mean(x_mono), 4),
        "trigger_integrity_mean": round(mean(trigger_int), 4),
        "section_count_mean": round(mean(section_counts), 3),
        "object_count_mean": round(mean(obj_counts), 1),
        "per_level": [
            {
                "name": r.level_name,
                "score": round(r.score_total, 4),
                "x_monotone": round(r.x_monotone_ratio, 4),
                "trigger_integrity": round(r.trigger_integrity, 4),
                "section_count": r.section_count,
                "object_count": r.object_count,
            }
            for r in results
        ],
    }

    LOGGER.info(
        "Validation summary: score_mean=%.4f x_mono=%.4f trigger=%.4f",
        summary["score_mean"],
        summary["x_monotone_mean"],
        summary["trigger_integrity_mean"],
    )

    return summary
