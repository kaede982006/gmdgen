# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

"""Tests for Phase 6: Candidate generation must produce distinct candidates.

The previous failure mode was candidate_count=3 producing identical candidates
because deterministic materialization used the same seed for every candidate.
The fix offsets the materialization seed by candidate_id."""

from dataclasses import replace as dc_replace

from gmdgen.gd.plans import SectionPlan
from gmdgen.gd.time_mapping import SpeedState
from gmdgen.generate.materializer import MaterializationConfig, materialize_level_plans


def _sections(n: int = 4) -> list[SectionPlan]:
    return [
        SectionPlan(
            start_time=i * 5.0, end_time=(i + 1) * 5.0,
            start_x=i * 1000.0, end_x=(i + 1) * 1000.0,
            section_type="verse", gameplay_mode="cube",
            speed_state=SpeedState.NORMAL,
            density_target=0.6, decoration_intensity=0.7,
            trigger_intensity=0.4, difficulty_target=0.5,
        )
        for i in range(n)
    ]


def _signature(objs) -> tuple:
    """Compact signature of generated objects for distinctness comparison."""
    return tuple((o.object_id, round(o.x, 1), round(o.y, 1)) for o in objs[:50])


def test_same_seed_produces_identical_output():
    """Determinism baseline: same config produces identical output."""
    config = MaterializationConfig(seed=42, target_object_count=200)
    sections = _sections()
    a = materialize_level_plans(sections, config=config, total_object_budget=200)
    b = materialize_level_plans(sections, config=config, total_object_budget=200)
    assert _signature(a) == _signature(b), "Deterministic baseline broken"


def test_different_seeds_produce_distinct_output():
    """Two different seeds must produce different signatures."""
    sections = _sections()
    base = MaterializationConfig(seed=42, target_object_count=200)
    candidate1 = dc_replace(base, seed=base.seed + 1 * 7919)
    candidate2 = dc_replace(base, seed=base.seed + 2 * 7919)
    candidate3 = dc_replace(base, seed=base.seed + 3 * 7919)

    out1 = materialize_level_plans(sections, config=candidate1, total_object_budget=200)
    out2 = materialize_level_plans(sections, config=candidate2, total_object_budget=200)
    out3 = materialize_level_plans(sections, config=candidate3, total_object_budget=200)

    sigs = {_signature(out1), _signature(out2), _signature(out3)}
    assert len(sigs) >= 2, (
        f"Three candidate seeds collapsed to {len(sigs)} signature(s); candidate variation broken"
    )


def test_candidate_seed_offset_pattern_used_in_audio_conditioned():
    """The audio_conditioned pipeline must offset seed by candidate_id."""
    from pathlib import Path
    text = Path("src/gmdgen/generate/audio_conditioned.py").read_text(encoding="utf-8")
    # The fix introduces `candidate_mat_config` and seeds it via candidate_id.
    assert "candidate_mat_config" in text, (
        "audio_conditioned.py must offset materialization seed per candidate"
    )
    assert "candidate_id" in text


def test_candidate_object_counts_vary_or_match_intentionally():
    """Different seeds may produce different counts; if they match, output still differs."""
    sections = _sections()
    candidates = [
        dc_replace(MaterializationConfig(seed=10, target_object_count=300), seed=10 + i * 7919)
        for i in range(4)
    ]
    outputs = [materialize_level_plans(sections, config=c, total_object_budget=300) for c in candidates]
    sigs = {_signature(o) for o in outputs}
    assert len(sigs) >= 2, "Candidates collapsed to identical output"


def test_candidate_signatures_are_recordable():
    """Signatures must be hashable for inclusion in candidate_reports."""
    sections = _sections()
    config = MaterializationConfig(seed=42, target_object_count=100)
    objs = materialize_level_plans(sections, config=config, total_object_budget=100)
    sig = _signature(objs)
    # Must be hashable
    assert hash(sig) is not None
    # Must be JSON-serializable in some form (we wrap as list of lists)
    import json
    serializable = [list(item) for item in sig]
    json.dumps(serializable)
