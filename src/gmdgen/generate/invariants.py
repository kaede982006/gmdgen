# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 kaede982006
"""Structural invariants enforced after generation.

If any invariant fails the candidate is **rejected**, never repaired into
existence. This is the explicit fix for the prior "spray" failure mode where
a repair loop quietly filled an empty canvas with 46258 random objects.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable


# Hard limits.
MIN_TOTAL_OBJECTS = 50
MAX_TOTAL_OBJECTS = 12000
MIN_TRIGGERS_PER_SECTION = 1
MIN_TRIGGERS_FLOOR = 3
MIN_GROUND_COVERAGE = 0.70
MIN_JUMPABLE_PATH = 0.95
MIN_UNIQUE_OBJECT_TYPES = 6


@dataclass(slots=True)
class InvariantResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass(slots=True)
class InvariantReport:
    results: list[InvariantResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> list[InvariantResult]:
        return [r for r in self.results if not r.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "results": [
                {"name": r.name, "passed": r.passed, "detail": r.detail}
                for r in self.results
            ],
        }


def _count_objects(objects: Iterable[Any]) -> int:
    if hasattr(objects, "__len__"):
        return len(objects)  # type: ignore[arg-type]
    return sum(1 for _ in objects)


def _is_trigger(obj: Any) -> bool:
    role = getattr(obj, "role", None)
    if isinstance(role, str) and "trigger" in role.lower():
        return True
    obj_id = str(getattr(obj, "object_id", "") or "")
    # Heuristic: GD trigger object IDs are typically ≥899 or in known ranges.
    if obj_id.isdigit() and int(obj_id) >= 899:
        return True
    return False


def _ground_coverage(objects: list[Any], section_count: int) -> float:
    """Fraction of beat-bins that contain at least one ground-level object."""
    if not objects:
        return 0.0
    ground = [o for o in objects if abs(getattr(o, "y", 999) - 105) <= 35]
    if not ground:
        return 0.0
    bins = Counter(int(getattr(o, "x", 0) // 30) for o in ground)
    if not bins:
        return 0.0
    total_bins = max(bins) - min(bins) + 1 if bins else 1
    filled = sum(1 for v in bins.values() if v > 0)
    return filled / max(1, total_bins)


def _unique_object_types(objects: list[Any]) -> int:
    return len({str(getattr(o, "object_id", "")) for o in objects if getattr(o, "object_id", "")})


def check_invariants(
    objects: list[Any],
    *,
    section_count: int = 1,
    triggers: list[Any] | None = None,
    jumpable_path_ratio: float | None = None,
) -> InvariantReport:
    """Return a structured report; never raises."""
    triggers = triggers or [o for o in objects if _is_trigger(o)]
    n_obj = _count_objects(objects)
    n_trig = _count_objects(triggers)
    coverage = _ground_coverage(objects, section_count)
    types = _unique_object_types(objects)
    jumpable = jumpable_path_ratio if jumpable_path_ratio is not None else 1.0

    results: list[InvariantResult] = []

    # I-1: object count band.
    if MIN_TOTAL_OBJECTS <= n_obj <= MAX_TOTAL_OBJECTS:
        results.append(InvariantResult("I-1.object_count_in_band", True, f"n={n_obj}"))
    else:
        results.append(InvariantResult(
            "I-1.object_count_in_band", False,
            f"n={n_obj} not in [{MIN_TOTAL_OBJECTS}, {MAX_TOTAL_OBJECTS}]",
        ))

    # I-2: trigger floor.
    expected = max(MIN_TRIGGERS_FLOOR, section_count * MIN_TRIGGERS_PER_SECTION)
    results.append(InvariantResult(
        "I-2.trigger_floor",
        n_trig >= expected,
        f"triggers={n_trig} expected>={expected}",
    ))

    # I-3: ground coverage.
    results.append(InvariantResult(
        "I-3.ground_line_coverage",
        coverage >= MIN_GROUND_COVERAGE,
        f"coverage={coverage:.3f}",
    ))

    # I-4: jumpable path.
    results.append(InvariantResult(
        "I-4.jumpable_path_ratio",
        jumpable >= MIN_JUMPABLE_PATH,
        f"jumpable={jumpable:.3f}",
    ))

    # I-5: object type variety.
    results.append(InvariantResult(
        "I-5.unique_object_types",
        types >= MIN_UNIQUE_OBJECT_TYPES,
        f"types={types} expected>={MIN_UNIQUE_OBJECT_TYPES}",
    ))

    return InvariantReport(results=results)


class InvariantViolation(RuntimeError):
    """Raised when a candidate fails one or more structural invariants."""

    def __init__(self, report: InvariantReport) -> None:
        self.report = report
        names = [r.name for r in report.failures]
        super().__init__(f"invariant_violation: {names}")


def assert_invariants(
    objects: list[Any],
    *,
    section_count: int = 1,
    triggers: list[Any] | None = None,
    jumpable_path_ratio: float | None = None,
) -> InvariantReport:
    rep = check_invariants(
        objects,
        section_count=section_count,
        triggers=triggers,
        jumpable_path_ratio=jumpable_path_ratio,
    )
    if not rep.passed:
        raise InvariantViolation(rep)
    return rep


def must_not_be_empty(raw_objects: int) -> None:
    """R0 from the v2.3 repair policy: empty canvas must not be filled."""
    if raw_objects == 0:
        raise InvariantViolation(InvariantReport(results=[
            InvariantResult("R0.raw_objects_must_be_nonzero", False, "raw_objects==0"),
        ]))
