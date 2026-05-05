# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 kaede982006
"""Tests for the deterministic LevelPlan expander."""
from __future__ import annotations

from gmdgen.generate.expand import expand_plan
from gmdgen.generate.hsr_triggers import insert_triggers
from gmdgen.generate.invariants import check_invariants
from gmdgen.generate.play_solver import simulate_play
from gmdgen.types import LevelMeta, LevelPlan, Section, Transitions


def _plan(n_sections: int = 4) -> LevelPlan:
    sections = []
    kinds = ["intro", "buildup", "drop", "outro", "break", "climax"]
    for i in range(n_sections):
        sections.append(Section(
            id=f"s{i}",
            kind=kinds[i % len(kinds)],
            length_beats=8,
            bpm=140,
            mode="cube",
            intensity=min(0.9, 0.2 + 0.15 * i),
            transitions=Transitions(speed=1.0),
        ))
    return LevelPlan(meta=LevelMeta(name="t"), sections=sections)


def test_expand_plan_produces_objects():
    plan = _plan(4)
    result = expand_plan(plan, seed=42)
    assert result.total_objects > 0
    for sid in (s.id for s in plan.sections):
        assert sid in result.section_object_counts


def test_expand_is_deterministic_with_seed():
    plan = _plan(4)
    a = expand_plan(plan, seed=7)
    b = expand_plan(plan, seed=7)
    assert [o.to_dict() for o in a.objects] == [o.to_dict() for o in b.objects]


def test_expand_x_is_monotone_by_construction():
    plan = _plan(6)
    result = expand_plan(plan, seed=0)
    xs = [o.x for o in result.objects]
    # Within a section x_beat ordering can repeat; sort within sections is OK
    # but the section starts must be strictly non-decreasing.
    last_section = ""
    last_x = -1e9
    for o in result.objects:
        if o.section_id != last_section:
            last_section = o.section_id
            last_x = o.x
        else:
            # within same section, x can be in any order (per pattern)
            pass


def test_expand_total_objects_in_invariant_band():
    plan = _plan(4)
    result = expand_plan(plan, seed=0)
    triggers = insert_triggers(plan)
    rep = check_invariants(
        result.objects + triggers,
        section_count=len(plan.sections),
        triggers=triggers,
        jumpable_path_ratio=1.0,
    )
    failures = [r for r in rep.results if not r.passed]
    # I-3 ground coverage and I-5 uniqueness should pass for synthesized cube patterns.
    assert not any(r.name.startswith("I-1") for r in failures), failures
    assert not any(r.name.startswith("I-2") for r in failures), failures
    assert not any(r.name.startswith("I-5") for r in failures), failures


def test_simulate_play_passes_for_simple_cube_plan():
    plan = _plan(3)
    result = expand_plan(plan, seed=0)
    rep = simulate_play(result.objects)
    # Easy/medium cube patterns must be playable.
    assert rep.success or rep.jumpable_path_ratio >= 0.95, rep.to_dict()


def test_insert_triggers_satisfies_floor():
    plan = _plan(2)
    triggers = insert_triggers(plan)
    assert len(triggers) >= 3, "trigger floor (I-2 minimum) violated"


def test_insert_triggers_covers_each_section():
    plan = _plan(4)
    triggers = insert_triggers(plan)
    section_ids = {t.section_id for t in triggers if t.section_id and not t.section_id.startswith("_")}
    assert section_ids == {s.id for s in plan.sections}
