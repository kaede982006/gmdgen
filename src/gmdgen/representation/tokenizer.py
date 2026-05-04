# SPDX-License-Identifier: GPL-3.0-or-later
"""Feature-aware tokeniser for GD level objects.

Goodfellow Ch.15 §15.3 "Semi-Supervised Disentangling of Causal Factors":
  Instead of a single scalar token OBJ:{id}, each object is encoded as a
  composite token that separates independent *factors of variation*:
    - object identity  (id)
    - spatial position (dx bucket — relative X distance from previous object)
    - vertical band    (y bucket — course height zone)
    - semantic class   (S/D/T/P/X/U)
    - section          (which speed/gamemode section the object is in)

Legacy OBJ:{id} tokens are still produced by `level_data_to_tokens()` for
backward-compatibility with the existing MarkovModel pipeline.
The new `level_data_to_feature_tokens()` produces richer tokens.
"""

from __future__ import annotations

import math

from gmdgen.data.preprocess import (
    detect_section_boundaries,
    split_level_objects,
)
from gmdgen.data.schema import GMDRecord
from gmdgen.features.tokenizer import (
    EOS_TOKEN,
    extract_object_id,
    extract_object_number,
)
from gmdgen.representation.object_classifier import class_short

# DX buckets — relative X distance to previous object, log-scaled
_DX_BINS = [0, 5, 15, 30, 60, 120, 300, 750]

# Y bands — absolute Y position mapped to 8 vertical zones
# GD screen height spans roughly −400 to +900 (centred at 180)
_Y_BINS = [-400, -150, 0, 100, 180, 300, 500, 900]


def _dx_bucket(dx: float) -> str:
    for i, threshold in enumerate(_DX_BINS[1:], start=1):
        if dx < threshold:
            return str(i - 1)
    return str(len(_DX_BINS) - 1)


def _y_bucket(y: float) -> str:
    for i, threshold in enumerate(_Y_BINS[1:], start=1):
        if y < threshold:
            return str(i - 1)
    return str(len(_Y_BINS) - 1)


def to_feature_token(
    level_object: str,
    *,
    previous_x: float | None,
    section_id: int,
) -> str | None:
    """Encode one GD object string as a feature token.

    Format:
        OBJ:{id}|CLS:{S/D/T/P/X/U}|DX:{bucket}|Y:{bucket}|SEC:{section_id}

    Returns None if the object has no valid id/x/y.
    """
    object_id = extract_object_id(level_object)
    if not object_id:
        return None

    x_val = extract_object_number(level_object, "2")
    y_val = extract_object_number(level_object, "3")
    if x_val is None or y_val is None:
        return None

    cls = class_short(object_id)
    dx = (x_val - previous_x) if previous_x is not None else 0.0
    dx = max(0.0, dx)
    dx_b = _dx_bucket(dx)
    y_b = _y_bucket(y_val)

    return f"OBJ:{object_id}|CLS:{cls}|DX:{dx_b}|Y:{y_b}|SEC:{section_id}"


def level_data_to_feature_tokens(level_data: str) -> list[str]:
    """Convert raw level data string to a list of feature tokens."""
    objects = split_level_objects(level_data)
    if not objects:
        return [EOS_TOKEN]

    boundaries = detect_section_boundaries(objects)
    # Build index → section_id map
    section_starts = sorted(b.start_object_index for b in boundaries)

    def _section_id(obj_index: int) -> int:
        sid = 0
        for idx, start in enumerate(section_starts):
            if obj_index >= start:
                sid = idx
        return sid

    tokens: list[str] = []
    previous_x: float | None = None

    for obj_index, obj in enumerate(objects):
        sec_id = _section_id(obj_index)
        token = to_feature_token(obj, previous_x=previous_x, section_id=sec_id)
        if token:
            tokens.append(token)
            x_val = extract_object_number(obj, "2")
            if x_val is not None:
                previous_x = x_val

    tokens.append(EOS_TOKEN)
    return tokens


def records_to_feature_sequences(records: list[GMDRecord]) -> list[list[str]]:
    return [
        level_data_to_feature_tokens(record.decoded_level_data)
        for record in records
    ]
