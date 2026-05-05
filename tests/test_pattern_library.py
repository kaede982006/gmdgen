# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 kaede982006
"""Pattern library coverage and shape."""
from __future__ import annotations

import json
import random
from pathlib import Path

from gmdgen.patterns.builder import (
    PATTERNS_INDEX_PATH,
    PATTERNS_PER_CELL,
    build_index,
    load_index,
    pick_pattern,
)
from gmdgen.types import VALID_DIFFICULTIES, VALID_GAME_MODES


def test_build_index_writes_all_cells(tmp_path: Path) -> None:
    out = tmp_path / "patterns_index.json"
    idx = build_index(out)
    assert out.exists()
    cells = idx["cells"]
    expected = {f"{m}/{d}" for m in VALID_GAME_MODES for d in VALID_DIFFICULTIES}
    assert set(cells.keys()) == expected, "every (mode, difficulty) cell must be represented"


def test_each_cell_has_minimum_patterns() -> None:
    idx = load_index()
    for cell, ids in idx["cells"].items():
        assert len(ids) >= PATTERNS_PER_CELL, (
            f"cell {cell} has {len(ids)} patterns; expected ≥ {PATTERNS_PER_CELL}"
        )


def test_no_cell_is_empty_stop_condition() -> None:
    """v2.3 STOP condition: empty mode×difficulty cell -> halt."""
    idx = load_index()
    for cell, ids in idx["cells"].items():
        assert ids, f"cell {cell} is empty — would trigger v2.3 STOP"


def test_total_patterns_at_least_100() -> None:
    """v2.3 instruction: ~100 patterns minimum."""
    idx = load_index()
    assert len(idx["patterns"]) >= 100, f"only {len(idx['patterns'])} patterns indexed"


def test_each_pattern_has_required_fields() -> None:
    idx = load_index()
    required = {"id", "mode", "difficulty", "length_beats", "objects", "entry", "exit"}
    for pid, pat in idx["patterns"].items():
        missing = required - set(pat)
        assert not missing, f"pattern {pid} missing fields: {missing}"


def test_pattern_objects_use_beat_relative_x() -> None:
    idx = load_index()
    for pat in list(idx["patterns"].values())[:30]:
        for obj in pat["objects"]:
            assert "x_beat" in obj, f"object missing x_beat in pattern {pat['id']}"
            assert 0 <= obj["x_beat"] <= pat["length_beats"]


def test_pick_pattern_deterministic_with_seed() -> None:
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    p1 = pick_pattern(mode="cube", difficulty="medium", rng=rng1)
    p2 = pick_pattern(mode="cube", difficulty="medium", rng=rng2)
    assert p1["id"] == p2["id"]


def test_pick_pattern_falls_back_for_unknown_cell() -> None:
    # Even if difficulty isn't recognized, we should not raise.
    rng = random.Random(0)
    pat = pick_pattern(mode="cube", difficulty="bogus", rng=rng)
    assert pat["mode"] in {"cube", "ship", "ball", "ufo", "wave", "robot", "spider"}


def test_index_file_is_valid_json() -> None:
    text = PATTERNS_INDEX_PATH.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert "cells" in payload and "patterns" in payload
