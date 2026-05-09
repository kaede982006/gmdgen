# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 kaede982006
"""Synthetic pattern generator + index builder.

Every (mode, difficulty) cell of the pattern library is populated with at
least six deterministic patterns. Each pattern is a beat-relative sequence
of object placements that the deterministic expander can splice together.

The synthesis is **not** a substitute for hand-tuned patterns; it provides
a defensible default so the library is never empty.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gmdgen.types import VALID_DIFFICULTIES, VALID_GAME_MODES


PATTERNS_DIR = Path(__file__).parent / "_data"
PATTERNS_INDEX_PATH = PATTERNS_DIR / "patterns_index.json"

# Object IDs known to be safe gameplay/decoration in Geometry Dash.
GAMEPLAY_BLOCK_IDS = ("1", "2", "3", "4", "5", "6")
GAMEPLAY_SPIKE_IDS = ("8", "9", "39", "103")
ORB_IDS = ("36", "84", "141", "1332", "1333")
PAD_IDS = ("35", "67", "140")
DECORATION_IDS = ("211", "503", "1734", "1736", "259", "266")

# Y-bands.
GROUND_Y = 105
LOW_Y = 135
MID_Y = 195
HIGH_Y = 255

PATTERNS_PER_CELL = 6
BEAT_UNIT = 30.0  # x units per beat at speed=1
REGENERATE_PATTERN_FIXTURES_ENV = "GMDGEN_REGENERATE_PATTERN_FIXTURES"


@dataclass(slots=True)
class Pattern:
    id: str
    mode: str
    difficulty: str
    length_beats: int
    objects: list[dict[str, Any]] = field(default_factory=list)
    entry: dict[str, Any] = field(default_factory=lambda: {"x_beat": 0.0, "y": GROUND_Y, "speed": 1.0})
    exit: dict[str, Any] = field(default_factory=lambda: {"x_beat": 8.0, "y": GROUND_Y, "speed": 1.0})
    tested: bool = True
    source: str = "synthetic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _density_for(difficulty: str) -> int:
    # Capped so hazard spacing in an 8-beat pattern is >= 2.5 beats (75u),
    # safely above play_solver.MIN_REACTION_X (60u).
    return {"easy": 2, "medium": 3, "hard": 3}.get(difficulty, 2)


def _synthesize(mode: str, difficulty: str, idx: int) -> Pattern:
    """Produce one deterministic pattern for the given cell."""
    rng = random.Random(hash((mode, difficulty, idx)) & 0xFFFF_FFFF)
    length_beats = 8
    density = _density_for(difficulty)
    objects: list[dict[str, Any]] = []

    # Always lay a ground rail so the section has structure.
    for b in range(length_beats):
        objects.append({
            "id": rng.choice(GAMEPLAY_BLOCK_IDS),
            "x_beat": float(b),
            "y": GROUND_Y,
            "role": "structural",
        })

    # Mode-specific gameplay objects.
    if mode == "cube":
        # Place gameplay objects at well-spaced beat slots so the play_solver's
        # MIN_REACTION_X (60u = 2 beats at speed=1) is always respected.
        for k in range(density):
            beat = (k + 1) * (length_beats / (density + 1))
            if difficulty == "easy":
                objects.append({"id": rng.choice(ORB_IDS), "x_beat": round(beat, 2), "y": LOW_Y, "role": "gameplay"})
            elif difficulty == "medium":
                objects.append({"id": rng.choice(GAMEPLAY_SPIKE_IDS), "x_beat": round(beat, 2), "y": GROUND_Y + 30, "role": "gameplay"})
            else:
                # hard: orb high, single spike. Avoid stacking spikes within 2 beats.
                objects.append({"id": rng.choice(ORB_IDS), "x_beat": round(beat, 2), "y": MID_Y, "role": "gameplay"})
    elif mode == "ship":
        # ceiling and floor bands
        for b in range(length_beats):
            objects.append({"id": rng.choice(GAMEPLAY_BLOCK_IDS), "x_beat": float(b), "y": HIGH_Y, "role": "structural"})
        for k in range(density):
            beat = k * (length_beats / density)
            objects.append({"id": rng.choice(GAMEPLAY_SPIKE_IDS), "x_beat": round(beat + 0.5, 2), "y": MID_Y, "role": "gameplay"})
    elif mode == "ball":
        for k in range(density):
            beat = k * 2.0
            objects.append({"id": rng.choice(PAD_IDS), "x_beat": round(beat, 2), "y": GROUND_Y, "role": "gameplay"})
    elif mode == "ufo":
        for k in range(density):
            beat = k * 1.5
            objects.append({"id": rng.choice(ORB_IDS), "x_beat": round(beat, 2), "y": MID_Y, "role": "gameplay"})
    elif mode == "wave":
        # tight corridor
        for b in range(length_beats):
            y = HIGH_Y if b % 2 == 0 else LOW_Y
            objects.append({"id": rng.choice(GAMEPLAY_BLOCK_IDS), "x_beat": float(b), "y": y, "role": "structural"})
    elif mode == "robot":
        for k in range(density):
            beat = k * 2.0
            objects.append({"id": rng.choice(ORB_IDS), "x_beat": round(beat + 0.5, 2), "y": LOW_Y, "role": "gameplay"})
    elif mode == "spider":
        for k in range(density):
            beat = k * 1.0
            objects.append({"id": rng.choice(PAD_IDS), "x_beat": round(beat, 2), "y": GROUND_Y, "role": "gameplay"})

    # Decoration sprinkles (kept above gameplay corridor).
    for _ in range(2):
        objects.append({
            "id": rng.choice(DECORATION_IDS),
            "x_beat": round(rng.uniform(0, length_beats - 0.5), 2),
            "y": rng.choice([HIGH_Y + 30, HIGH_Y + 60]),
            "role": "decoration",
        })

    return Pattern(
        id=f"{mode}-{difficulty}-{idx:02d}",
        mode=mode,
        difficulty=difficulty,
        length_beats=length_beats,
        objects=objects,
        entry={"x_beat": 0.0, "y": GROUND_Y, "speed": 1.0},
        exit={"x_beat": float(length_beats), "y": GROUND_Y, "speed": 1.0},
    )


def build_index(
    out_path: Path | None = None,
    *,
    patterns_dir: Path | None = None,
    write_pattern_files: bool | None = None,
) -> dict[str, Any]:
    """Generate all (mode, difficulty) patterns and optionally write fixtures.

    Tests should call this with a temporary ``out_path`` and must not rewrite
    package fixtures. Source fixture regeneration is opt-in via
    ``GMDGEN_REGENERATE_PATTERN_FIXTURES=1`` or an explicit
    ``write_pattern_files=True`` maintenance call.
    """
    target = out_path or PATTERNS_INDEX_PATH
    target_dir = patterns_dir or (PATTERNS_DIR if target == PATTERNS_INDEX_PATH else target.parent)
    regenerate_enabled = _fixture_regeneration_enabled()
    should_write_pattern_files = regenerate_enabled if write_pattern_files is None else bool(write_pattern_files)

    if should_write_pattern_files:
        target_dir.mkdir(parents=True, exist_ok=True)
    elif target.parent != PATTERNS_DIR or out_path is not None:
        target.parent.mkdir(parents=True, exist_ok=True)

    index: dict[str, Any] = {"version": 1, "cells": {}, "patterns": {}}

    for mode in VALID_GAME_MODES:
        for difficulty in VALID_DIFFICULTIES:
            cell_dir = target_dir / mode / difficulty
            if should_write_pattern_files:
                cell_dir.mkdir(parents=True, exist_ok=True)
            cell_ids: list[str] = []
            for i in range(PATTERNS_PER_CELL):
                pat = _synthesize(mode, difficulty, i)
                if should_write_pattern_files:
                    fp = cell_dir / f"{pat.id}.json"
                    fp.write_text(
                        json.dumps(pat.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                cell_ids.append(pat.id)
                index["patterns"][pat.id] = pat.to_dict()
            index["cells"][f"{mode}/{difficulty}"] = cell_ids

    if out_path is not None or regenerate_enabled:
        target.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def load_index(path: Path | None = None) -> dict[str, Any]:
    p = path or PATTERNS_INDEX_PATH
    if not p.exists():
        return build_index(p if _fixture_regeneration_enabled() or path is not None else None)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return build_index(p if _fixture_regeneration_enabled() or path is not None else None)


def _fixture_regeneration_enabled() -> bool:
    return os.environ.get(REGENERATE_PATTERN_FIXTURES_ENV, "").strip() in {"1", "true", "yes"}


def pick_pattern(
    *,
    mode: str,
    difficulty: str,
    rng: random.Random,
    index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    idx = index or load_index()
    cell_key = f"{mode}/{difficulty}"
    cell_ids = idx.get("cells", {}).get(cell_key) or []
    if not cell_ids:
        # Fallback: any pattern with same mode, then anything.
        for k, v in idx.get("cells", {}).items():
            if k.startswith(f"{mode}/") and v:
                cell_ids = v
                break
    if not cell_ids:
        cell_ids = ["cube-easy-00"]
    chosen_id = rng.choice(cell_ids)
    return idx["patterns"][chosen_id]


if __name__ == "__main__":
    idx = build_index()
    cells = len(idx["cells"])
    pats = len(idx["patterns"])
    print(f"built {cells} cells, {pats} patterns")
