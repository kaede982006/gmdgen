# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 kaede982006
"""Factorized tokenization for the GMD language model.

Following Goodfellow Ch.15 (disentangling causal factors) and Fleuret §4.9
(token embedding), each GD object is encoded as several independent
sub-tokens. The model learns one embedding table per factor, and the input
representation is the sum of factor embeddings — this is what gives the
network statistical strength across rare combinations.

Factors:
  * id        — object_id mapped to a compact id space (top-K + UNK).
  * cls       — semantic class: STRUCTURE/DECORATION/TRIGGER/PORTAL/SPECIAL/UNK.
  * dx        — log-spaced bin of relative X distance from previous object.
  * y         — bin of absolute Y position.
  * mode      — game mode index (cube/ship/ball/ufo/wave/robot/spider).
  * speed     — speed-portal state (0.5x..4x).
  * section   — section index within a level.

Special token ids:
  * BOS_ID, EOS_ID, PAD_ID, UNK_ID are reserved at the *start* of the id
    vocabulary so that 0 is always padding.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from gmdgen.data.preprocess import detect_section_boundaries, split_level_objects
from gmdgen.features.tokenizer import extract_object_id, extract_object_number
from gmdgen.representation.object_classifier import (
    ObjectClass,
    classify,
)

# ── special slots ────────────────────────────────────────────────────────────
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
NUM_SPECIAL = 4

# ── bins ─────────────────────────────────────────────────────────────────────
DX_BINS = (0.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 750.0, 1e9)
Y_BINS = (-400.0, -150.0, 0.0, 100.0, 180.0, 300.0, 500.0, 900.0, 1e9)

# ── enums ────────────────────────────────────────────────────────────────────
CLS_ORDER = (
    ObjectClass.UNKNOWN,
    ObjectClass.STRUCTURE,
    ObjectClass.DECORATION,
    ObjectClass.TRIGGER,
    ObjectClass.PORTAL,
    ObjectClass.SPECIAL,
)
CLS_TO_INDEX = {c: i for i, c in enumerate(CLS_ORDER)}

MODE_NAMES = ("cube", "ship", "ball", "ufo", "wave", "robot", "spider")
SPEED_NAMES = ("0.5x", "1x", "2x", "3x", "4x")
DEFAULT_MODE_IDX = 0  # cube
DEFAULT_SPEED_IDX = 1  # 1x

# ── object id vocabulary ─────────────────────────────────────────────────────
# Top-K most common ids in the training data go into the slots
# [NUM_SPECIAL, NUM_SPECIAL+TOP_K). Anything else collapses to UNK_ID.
TOP_K_IDS = 256
ID_VOCAB_SIZE = NUM_SPECIAL + TOP_K_IDS

# How many bins per factor (after the special prefix is *not* added — these
# are factor-internal vocabularies; the final input embedding sums them with
# the id embedding).
DX_VOCAB = len(DX_BINS)            # 9 bins (8 thresholds + overflow)
Y_VOCAB = len(Y_BINS)              # 9 bins
CLS_VOCAB = len(CLS_ORDER)         # 6
MODE_VOCAB = len(MODE_NAMES)       # 7
SPEED_VOCAB = len(SPEED_NAMES)     # 5
SECTION_VOCAB = 32                 # absolute cap, sections beyond clamp to last


# Mode/speed transitions: portal id → mode/speed delta. We rely on the
# classifier sets but expose explicit mappings so the encoder can advance the
# running state token-by-token.
_MODE_PORTAL = {
    "12": 0,    # cube
    "13": 1,    # ship
    "47": 2,    # ball
    "111": 3,   # ufo
    "660": 4,   # wave
    "745": 5,   # robot
    "1331": 6,  # spider
}
_SPEED_PORTAL = {
    "200": 0,   # 0.5x
    "201": 1,   # 1x
    "202": 2,   # 2x
    "203": 3,   # 3x
    "1334": 4,  # 4x
}


# ── data classes ─────────────────────────────────────────────────────────────
@dataclass(slots=True)
class FactorizedToken:
    id_idx: int
    cls_idx: int
    dx_idx: int
    y_idx: int
    mode_idx: int
    speed_idx: int
    section_idx: int

    # optional originals — useful for round-tripping / sampling
    raw_object_id: str = ""
    raw_x: float = 0.0
    raw_y: float = 0.0


@dataclass(slots=True)
class IdVocab:
    """Mapping object_id (string) → id slot in [NUM_SPECIAL, ID_VOCAB_SIZE)."""

    id_to_slot: dict[str, int] = field(default_factory=dict)

    @property
    def vocab_size(self) -> int:
        return ID_VOCAB_SIZE

    def slot(self, object_id: str) -> int:
        return self.id_to_slot.get(object_id, UNK_ID)

    def to_dict(self) -> dict:
        return {"id_to_slot": dict(self.id_to_slot)}

    @classmethod
    def from_dict(cls, data: dict) -> "IdVocab":
        return cls(id_to_slot=dict(data.get("id_to_slot", {})))


def build_id_vocab(level_strings: Iterable[str]) -> IdVocab:
    """Pick the TOP_K_IDS most-frequent object ids and assign them slots.

    Special slots are reserved at the start of the id space.
    """
    from collections import Counter

    counts: Counter[str] = Counter()
    for s in level_strings:
        for obj in split_level_objects(s):
            oid = extract_object_id(obj)
            if oid:
                counts[oid] += 1
    top = counts.most_common(TOP_K_IDS)
    id_to_slot: dict[str, int] = {}
    for i, (oid, _) in enumerate(top):
        id_to_slot[oid] = NUM_SPECIAL + i
    return IdVocab(id_to_slot=id_to_slot)


# ── bin helpers ──────────────────────────────────────────────────────────────
def _bucket(value: float, edges: tuple[float, ...]) -> int:
    for i in range(len(edges) - 1):
        if value < edges[i + 1]:
            return i
    return len(edges) - 1


def dx_bucket(dx: float) -> int:
    return _bucket(max(0.0, dx), DX_BINS)


def y_bucket(y: float) -> int:
    return _bucket(y, Y_BINS)


def class_idx_for_object_id(object_id: str) -> int:
    """Return the semantic class factor used by the ML token stream."""
    return CLS_TO_INDEX.get(classify(object_id), 0)


def update_running_state(
    object_id: str,
    *,
    mode_idx: int,
    speed_idx: int,
) -> tuple[int, int]:
    """Advance mode/speed factors with the same state machine as encoding."""
    if object_id in _MODE_PORTAL:
        mode_idx = _MODE_PORTAL[object_id]
    if object_id in _SPEED_PORTAL:
        speed_idx = _SPEED_PORTAL[object_id]
    return mode_idx, speed_idx


def mode_portal_object_id(mode_idx: int) -> str | None:
    """Return the canonical portal id that enters a mode, if known."""
    for object_id, idx in _MODE_PORTAL.items():
        if idx == mode_idx:
            return object_id
    return None


# ── encode ───────────────────────────────────────────────────────────────────
def encode_level_string(level_data: str, vocab: IdVocab) -> list[FactorizedToken]:
    """Convert one decoded level data string into a list of FactorizedToken."""
    objects = split_level_objects(level_data)
    if not objects:
        return []

    boundaries = detect_section_boundaries(objects)
    section_starts = sorted(b.start_object_index for b in boundaries)

    def _section_id(obj_index: int) -> int:
        sid = 0
        for idx, start in enumerate(section_starts):
            if obj_index >= start:
                sid = idx
        return min(sid, SECTION_VOCAB - 1)

    tokens: list[FactorizedToken] = []
    previous_x: float | None = None
    mode_idx = DEFAULT_MODE_IDX
    speed_idx = DEFAULT_SPEED_IDX

    for i, obj in enumerate(objects):
        oid = extract_object_id(obj)
        if not oid:
            continue
        x_val = extract_object_number(obj, "2")
        y_val = extract_object_number(obj, "3")
        if x_val is None or y_val is None:
            continue

        mode_idx, speed_idx = update_running_state(
            oid,
            mode_idx=mode_idx,
            speed_idx=speed_idx,
        )

        cls_idx = class_idx_for_object_id(oid)
        dx = (x_val - previous_x) if previous_x is not None else 0.0
        tokens.append(
            FactorizedToken(
                id_idx=vocab.slot(oid),
                cls_idx=cls_idx,
                dx_idx=dx_bucket(dx),
                y_idx=y_bucket(y_val),
                mode_idx=mode_idx,
                speed_idx=speed_idx,
                section_idx=_section_id(i),
                raw_object_id=oid,
                raw_x=x_val,
                raw_y=y_val,
            )
        )
        previous_x = x_val

    return tokens


def tokens_to_id_array(tokens: list[FactorizedToken]) -> dict[str, list[int]]:
    """Stack token field arrays for tensor construction."""
    return {
        "id": [t.id_idx for t in tokens],
        "cls": [t.cls_idx for t in tokens],
        "dx": [t.dx_idx for t in tokens],
        "y": [t.y_idx for t in tokens],
        "mode": [t.mode_idx for t in tokens],
        "speed": [t.speed_idx for t in tokens],
        "section": [t.section_idx for t in tokens],
    }


__all__ = [
    "PAD_ID",
    "BOS_ID",
    "EOS_ID",
    "UNK_ID",
    "NUM_SPECIAL",
    "DX_BINS",
    "Y_BINS",
    "DX_VOCAB",
    "Y_VOCAB",
    "CLS_VOCAB",
    "MODE_VOCAB",
    "SPEED_VOCAB",
    "SECTION_VOCAB",
    "ID_VOCAB_SIZE",
    "TOP_K_IDS",
    "DEFAULT_MODE_IDX",
    "DEFAULT_SPEED_IDX",
    "FactorizedToken",
    "IdVocab",
    "build_id_vocab",
    "class_idx_for_object_id",
    "dx_bucket",
    "mode_portal_object_id",
    "update_running_state",
    "y_bucket",
    "encode_level_string",
    "tokens_to_id_array",
]
