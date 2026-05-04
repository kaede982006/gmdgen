from __future__ import annotations

from collections import Counter
from statistics import mean, stdev

from gmdgen.features.stats import summarize_sequences
from gmdgen.features.tokenizer import extract_object_id, extract_object_number


def _x_monotone_ratio(objects: list[str]) -> float:
    """Fraction of consecutive object pairs where x[i] >= x[i-1]."""
    prev: float | None = None
    ordered = 0
    total = 0
    for obj in objects:
        x_val = extract_object_number(obj, "2")
        if x_val is None:
            continue
        if prev is not None:
            total += 1
            if x_val >= prev:
                ordered += 1
        prev = x_val
    return round(ordered / total, 4) if total > 0 else 1.0


def _density_stats(objects: list[str], grid_unit: int = 30) -> dict[str, float]:
    """Per-grid-unit object density — mean and std."""
    bucket_counts: Counter[int] = Counter()
    for obj in objects:
        x_val = extract_object_number(obj, "2")
        if x_val is not None:
            bucket_counts[int(x_val) // grid_unit] += 1

    if not bucket_counts:
        return {"density_mean": 0.0, "density_std": 0.0}

    counts = list(bucket_counts.values())
    return {
        "density_mean": round(mean(counts), 3),
        "density_std": round(stdev(counts), 3) if len(counts) > 1 else 0.0,
    }


def evaluate_level_structure(decoded_level_data: str) -> dict[str, float | int]:
    """Compute structural quality metrics for a single decoded level string."""
    from gmdgen.data.preprocess import (
        detect_section_boundaries,
        split_level_objects,
    )

    objects = split_level_objects(decoded_level_data)
    if not objects:
        return {
            "object_count": 0,
            "x_monotone_ratio": 1.0,
            "section_count": 0,
            "density_mean": 0.0,
            "density_std": 0.0,
        }

    boundaries = detect_section_boundaries(objects)

    result: dict[str, float | int] = {
        "object_count": len(objects),
        "x_monotone_ratio": _x_monotone_ratio(objects),
        "section_count": len(boundaries),
    }
    result.update(_density_stats(objects))
    return result


def evaluate_training_run(
    *,
    total_records: int,
    used_records: int,
    sequences: list[list[str]],
) -> dict[str, float | int]:
    sequence_summary = summarize_sequences(sequences)
    usage_ratio = 0.0 if total_records == 0 else used_records / total_records

    return {
        "total_records": total_records,
        "used_records": used_records,
        "usage_ratio": round(usage_ratio, 4),
        **sequence_summary,
    }
