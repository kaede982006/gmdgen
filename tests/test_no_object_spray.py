# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 kaede982006
"""V'-2: regression — empty/oversized output must never reach disk.

These tests guard against the prior 46258-object spray failure mode by
verifying that the invariant layer aborts the candidate at four classic
attack vectors: empty AI response, oversized output, missing triggers,
and single-ID floods.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from gmdgen.generate.invariants import (
    InvariantViolation,
    MAX_TOTAL_OBJECTS,
    assert_invariants,
    must_not_be_empty,
)


@dataclass(slots=True)
class _Obj:
    object_id: str
    x: float
    y: float = 105
    role: str = "structural"


# ── attack: empty AI response ──────────────────────────────────────

def test_empty_raw_objects_aborts_save():
    """raw_objects==0 must raise; downstream save MUST NOT proceed."""
    with pytest.raises(InvariantViolation) as exc:
        must_not_be_empty(0)
    failed = exc.value.report.failures
    assert any("R0.raw_objects_must_be_nonzero" in r.name for r in failed)


# ── attack: oversized output (the 46258 case) ──────────────────────

def test_oversized_output_aborts():
    """final_objects > 12000 must abort, never reach the encoder."""
    flood = [_Obj(object_id=str(i % 30), x=float(i), y=105 + (i % 5))
             for i in range(MAX_TOTAL_OBJECTS + 100)]
    triggers = [_Obj(object_id="899", x=10.0, y=15, role="trigger") for _ in range(5)]
    with pytest.raises(InvariantViolation) as exc:
        assert_invariants(flood, section_count=3, triggers=triggers, jumpable_path_ratio=0.99)
    assert any(r.name.startswith("I-1") for r in exc.value.report.failures)


# ── attack: missing triggers ───────────────────────────────────────

def test_zero_triggers_aborts():
    objs = [_Obj(object_id=str((i % 6) + 1), x=i * 30.0, y=105) for i in range(200)]
    with pytest.raises(InvariantViolation) as exc:
        assert_invariants(objs, section_count=4, triggers=[], jumpable_path_ratio=0.99)
    assert any(r.name.startswith("I-2") for r in exc.value.report.failures)


# ── attack: single-ID flood (low diversity) ────────────────────────

def test_single_id_flood_aborts():
    objs = [_Obj(object_id="1", x=i * 30.0, y=105) for i in range(500)]
    triggers = [_Obj(object_id="899", x=10.0, y=15, role="trigger") for _ in range(3)]
    with pytest.raises(InvariantViolation) as exc:
        assert_invariants(objs, section_count=1, triggers=triggers, jumpable_path_ratio=0.99)
    assert any(r.name.startswith("I-5") for r in exc.value.report.failures)


# ── attack: skywriting (no ground objects) ─────────────────────────

def test_no_ground_coverage_aborts():
    objs = [_Obj(object_id=str(i % 6 + 1), x=i * 30.0, y=400) for i in range(200)]
    triggers = [_Obj(object_id="899", x=10.0, y=15, role="trigger") for _ in range(3)]
    with pytest.raises(InvariantViolation) as exc:
        assert_invariants(objs, section_count=1, triggers=triggers, jumpable_path_ratio=0.99)
    assert any(r.name.startswith("I-3") for r in exc.value.report.failures)


# ── attack: tight hazard pairs (jumpable_path_ratio simulation) ───

def test_unjumpable_path_aborts():
    objs = [_Obj(object_id=str(i % 6 + 1), x=i * 30.0, y=105) for i in range(200)]
    triggers = [_Obj(object_id="899", x=10.0, y=15, role="trigger") for _ in range(3)]
    with pytest.raises(InvariantViolation) as exc:
        assert_invariants(objs, section_count=1, triggers=triggers, jumpable_path_ratio=0.30)
    assert any(r.name.startswith("I-4") for r in exc.value.report.failures)


# ── property-style fuzzing using deterministic seeds ───────────────

@pytest.mark.parametrize("count,expected", [
    (0, "abort"),    # empty raw_objects
    (10, "abort"),   # too small
    (200, "ok"),     # in band
    (15000, "abort"),  # oversized
    (50000, "abort"),  # the 46258-style spray
])
def test_object_count_band_gating(count: int, expected: str):
    if count == 0:
        with pytest.raises(InvariantViolation):
            must_not_be_empty(0)
        return
    objs = [_Obj(object_id=str((i % 8) + 1), x=i * 30.0, y=105) for i in range(count)]
    triggers = [_Obj(object_id="899", x=10.0, y=15, role="trigger") for _ in range(5)]
    if expected == "abort":
        with pytest.raises(InvariantViolation):
            assert_invariants(objs, section_count=2, triggers=triggers, jumpable_path_ratio=0.99)
    else:
        assert_invariants(objs, section_count=2, triggers=triggers, jumpable_path_ratio=0.99)
