# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from gmdgen.data.preprocess import split_level_objects
from gmdgen.data.schema import GMDRecord

EOS_TOKEN = "<EOS>"


def extract_object_id(level_object: str) -> str | None:
    parts = [part.strip() for part in level_object.split(",")]
    if len(parts) < 2:
        return None

    if parts[0] == "1":
        return parts[1] or None

    for idx in range(0, len(parts) - 1, 2):
        if parts[idx] == "1":
            return parts[idx + 1] or None

    return None


def parse_object_pairs(level_object: str) -> list[tuple[str, str]]:
    parts = [part.strip() for part in level_object.split(",")]
    if len(parts) < 2:
        return []

    pairs: list[tuple[str, str]] = []
    for idx in range(0, len(parts) - 1, 2):
        key = parts[idx]
        value = parts[idx + 1]
        if key:
            pairs.append((key, value))
    return pairs


def _find_pair_index(pairs: list[tuple[str, str]], key: str) -> int | None:
    for index, (pair_key, _) in enumerate(pairs):
        if pair_key == key:
            return index
    return None


def extract_object_field(level_object: str, key: str) -> str | None:
    for pair_key, value in parse_object_pairs(level_object):
        if pair_key == key:
            return value
    return None


def extract_object_number(level_object: str, key: str) -> float | None:
    raw_value = extract_object_field(level_object, key)
    if raw_value is None:
        return None
    try:
        return float(raw_value)
    except ValueError:
        return None


def rewrite_object_xy(level_object: str, *, x: int, y: int) -> str:
    pairs = parse_object_pairs(level_object)
    if not pairs:
        object_id = extract_object_id(level_object) or "1"
        return f"1,{object_id},2,{x},3,{y}"

    x_index = _find_pair_index(pairs, "2")
    y_index = _find_pair_index(pairs, "3")

    if x_index is None:
        pairs.append(("2", str(x)))
    else:
        pairs[x_index] = ("2", str(x))

    if y_index is None:
        pairs.append(("3", str(y)))
    else:
        pairs[y_index] = ("3", str(y))

    flattened: list[str] = []
    for key, value in pairs:
        flattened.extend([key, value])
    return ",".join(flattened)


def level_data_to_tokens(level_data: str) -> list[str]:
    tokens: list[str] = []
    for level_object in split_level_objects(level_data):
        object_id = extract_object_id(level_object)
        if object_id:
            tokens.append(f"OBJ:{object_id}")

    tokens.append(EOS_TOKEN)
    return tokens


def records_to_token_sequences(records: list[GMDRecord]) -> list[list[str]]:
    return [level_data_to_tokens(record.decoded_level_data) for record in records]


def tokens_to_object_ids(tokens: list[str]) -> list[str]:
    object_ids: list[str] = []
    for token in tokens:
        if not token.startswith("OBJ:"):
            continue
        object_ids.append(token.split(":", maxsplit=1)[1])
    return object_ids
