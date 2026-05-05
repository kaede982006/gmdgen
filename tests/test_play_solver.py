# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 kaede982006
"""Tests for the heuristic play solver."""
from __future__ import annotations

from dataclasses import dataclass

from gmdgen.generate.play_solver import simulate_play


@dataclass(slots=True)
class _Obj:
    object_id: str
    x: float
    y: float = 105
    role: str = "structural"


def test_empty_returns_failure():
    rep = simulate_play([])
    assert rep.success is False
    assert rep.reason == "empty_object_list"


def test_simple_run_passes():
    objs = [_Obj(object_id=str((i % 6) + 1), x=i * 30.0, y=105) for i in range(50)]
    rep = simulate_play(objs)
    assert rep.success is True


def test_tight_hazard_pair_fails():
    """Two consecutive spikes within MIN_REACTION_X must be flagged."""
    objs = [
        _Obj(object_id="1", x=0.0, y=105, role="structural"),
        _Obj(object_id="8", x=100.0, y=105, role="gameplay"),
        _Obj(object_id="8", x=110.0, y=105, role="gameplay"),  # gap=10 < 60
    ]
    rep = simulate_play(objs)
    assert rep.success is False
    assert any("tight_hazard_pair" in i for i in rep.issues)


def test_out_of_reach_hazard_fails():
    objs = [
        _Obj(object_id="1", x=0.0, y=105, role="structural"),
        _Obj(object_id="8", x=200.0, y=400, role="gameplay"),  # too high
    ]
    rep = simulate_play(objs)
    assert rep.success is False
    assert any("out_of_reach" in i for i in rep.issues)


def test_jumpable_path_ratio_in_range():
    objs = [_Obj(object_id="1", x=i * 30.0, y=105) for i in range(20)]
    rep = simulate_play(objs)
    assert 0.0 <= rep.jumpable_path_ratio <= 1.0
