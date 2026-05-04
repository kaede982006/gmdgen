# SPDX-License-Identifier: GPL-3.0-or-later
"""Representation-level statistics for feature-token sequences.

Re-exports the legacy stats from gmdgen.features.stats and adds
richer structure-aware metrics.

Goodfellow Ch.11 §11.1 "Performance Metrics":
  A well-chosen metric should correlate with the actual task goal.
  For GD level generation the relevant axes are:
    - sequence diversity (unique token ratio)
    - object-class balance (structure / decoration / trigger proportions)
    - spatial regularity  (dx bucket distribution entropy)
    - section coverage    (how many distinct sections appear)
"""

from __future__ import annotations

import math
from collections import Counter

from gmdgen.features.stats import object_token_frequencies, summarize_sequences
from gmdgen.representation.object_classifier import ObjectClass, classify

__all__ = ["object_token_frequencies", "summarize_sequences", "summarize_feature_sequences"]


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum(
        (c / total) * math.log2(c / total)
        for c in counter.values()
        if c > 0
    )


def _extract_field(token: str, field: str) -> str | None:
    for part in token.split("|"):
        if part.startswith(field + ":"):
            return part[len(field) + 1:]
    return None


def summarize_feature_sequences(
    sequences: list[list[str]],
) -> dict[str, float | int]:
    if not sequences:
        return {}

    class_counts: Counter[str] = Counter()
    dx_counts: Counter[str] = Counter()
    sec_counts: Counter[str] = Counter()
    total_tokens = 0

    for seq in sequences:
        for token in seq:
            if not token.startswith("OBJ:"):
                continue
            total_tokens += 1

            obj_id_part = _extract_field(token, "OBJ") or ""
            cls = classify(obj_id_part).value if obj_id_part else ObjectClass.UNKNOWN.value
            class_counts[cls] += 1

            dx_b = _extract_field(token, "DX")
            if dx_b:
                dx_counts[dx_b] += 1

            sec = _extract_field(token, "SEC")
            if sec:
                sec_counts[sec] += 1

    if total_tokens == 0:
        return {"total_feature_tokens": 0}

    class_ratios = {
        f"class_ratio_{k}": round(v / total_tokens, 4)
        for k, v in class_counts.items()
    }

    return {
        "total_feature_tokens": total_tokens,
        "dx_entropy": round(_entropy(dx_counts), 4),
        "sec_count": len(sec_counts),
        "class_entropy": round(_entropy(class_counts), 4),
        **class_ratios,
    }
