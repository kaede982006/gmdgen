# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 kaede982006
"""Tests for structural invariants I-1..I-5 and the empty-canvas guard (R0)."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from gmdgen.generate.invariants import (
    InvariantViolation,
    MAX_TOTAL_OBJECTS,
    MIN_TOTAL_OBJECTS,
    assert_invariants,
    check_invariants,
    must_not_be_empty,
)


@dataclass(slots=True)
class _Obj:
    object_id: str
    x: float
    y: float = 105
    role: str = "structural"


def _ground_run(n: int) -> list[_Obj]:
    return [_Obj(object_id=str((i % 6) + 1), x=i * 30.0, y=105, role="structural")
            for i in range(n)]


def test_R0_empty_raw_aborts():
    with pytest.raises(InvariantViolation):
        must_not_be_empty(0)


def test_R0_nonempty_passes():
    must_not_be_empty(120)  # no exception


def test_I1_too_few_objects_fails():
    objs = _ground_run(MIN_TOTAL_OBJECTS - 5)
    rep = check_invariants(objs, section_count=1)
    assert any(r.name.startswith("I-1") and not r.passed for r in rep.results)


def test_I1_too_many_objects_fails():
    objs = _ground_run(MAX_TOTAL_OBJECTS + 10)
    rep = check_invariants(objs, section_count=1)
    assert any(r.name.startswith("I-1") and not r.passed for r in rep.results)


def test_I2_trigger_floor_required():
    objs = _ground_run(200)
    rep = check_invariants(objs, section_count=5, triggers=[])
    assert any(r.name.startswith("I-2") and not r.passed for r in rep.results)


def test_I3_no_ground_coverage_fails():
    # All objects above ground band → I-3 fails.
    objs = [_Obj(object_id=str(i % 5 + 1), x=i * 30.0, y=400) for i in range(200)]
    rep = check_invariants(objs, section_count=1)
    assert any(r.name.startswith("I-3") and not r.passed for r in rep.results)


def test_I4_jumpable_path_below_threshold_fails():
    objs = _ground_run(200)
    rep = check_invariants(objs, section_count=1, jumpable_path_ratio=0.5)
    assert any(r.name.startswith("I-4") and not r.passed for r in rep.results)


def test_I5_unique_object_types_floor():
    # Only 1 unique ID across many objects -> I-5 fails.
    objs = [_Obj(object_id="1", x=i * 30.0, y=105) for i in range(200)]
    rep = check_invariants(objs, section_count=1)
    assert any(r.name.startswith("I-5") and not r.passed for r in rep.results)


def test_passing_all_invariants():
    triggers = [_Obj(object_id="899", x=10.0, y=15, role="trigger") for _ in range(5)]
    objs = _ground_run(200) + triggers
    rep = check_invariants(objs, section_count=3, triggers=triggers, jumpable_path_ratio=0.99)
    # ground rail uses ids 1..6 → unique types satisfied; triggers cover I-2.
    assert rep.passed, [r for r in rep.results if not r.passed]


def test_assert_invariants_raises_on_violation():
    objs = [_Obj(object_id="1", x=0, y=999) for _ in range(10)]
    with pytest.raises(InvariantViolation):
        assert_invariants(objs, section_count=1)


def test_invariant_report_serializable():
    rep = check_invariants(_ground_run(10), section_count=1)
    d = rep.to_dict()
    import json
    json.dumps(d)
