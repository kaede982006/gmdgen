# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import pytest
from gmdgen.gd.plans import SectionPlan
from gmdgen.gd.time_mapping import SpeedState
from gmdgen.generate.materializer import MaterializationConfig, SectionObjectMaterializer, materialize_level_plans

def test_materializer_scaling():
    config = MaterializationConfig(object_multiplier=10.0)
    section = SectionPlan(
        start_time=0.0,
        end_time=10.0,
        start_x=0.0,
        end_x=1000.0,
        section_type="normal",
        gameplay_mode="cube",
        speed_state=SpeedState.NORMAL,
        density_target=0.5,
        decoration_intensity=0.5,
        trigger_intensity=0.5,
        difficulty_target=0.5,
    )
    
    materializer = SectionObjectMaterializer(config)
    objects = materializer.materialize_section(section, object_budget=5000)
    
    assert len(objects) > 10
    assert len(objects) < 5000

def test_materializer_determinism():
    config = MaterializationConfig(seed=42)
    section = SectionPlan(
        start_time=0.0,
        end_time=5.0,
        start_x=0.0,
        end_x=500.0,
        section_type="normal",
        gameplay_mode="cube",
        speed_state=SpeedState.NORMAL,
        density_target=0.5,
        decoration_intensity=0.5,
        trigger_intensity=0.5,
        difficulty_target=0.5,
    )
    
    materializer1 = SectionObjectMaterializer(config)
    objects1 = materializer1.materialize_section(section)
    
    materializer2 = SectionObjectMaterializer(config)
    objects2 = materializer2.materialize_section(section)
    
    assert len(objects1) == len(objects2)
    for o1, o2 in zip(objects1, objects2):
        assert o1.object_id == o2.object_id
        assert o1.x == o2.x
        assert o1.y == o2.y

def test_materialize_level_plans_budget():
    config = MaterializationConfig(object_multiplier=100.0) # High multiplier
    sections = [
        SectionPlan(
            start_time=0.0,
            end_time=5.0,
            start_x=0.0,
            end_x=500.0,
            section_type="normal",
            gameplay_mode="cube",
            speed_state=SpeedState.NORMAL,
            density_target=0.5,
            decoration_intensity=0.5,
            trigger_intensity=0.5,
            difficulty_target=0.5,
        )
    ]
    
    # Respect total budget
    objects = materialize_level_plans(sections, config=config, total_object_budget=100)
    assert len(objects) <= 100
    
    # Respect target_object_count
    config.target_object_count = 50
    objects = materialize_level_plans(sections, config=config, total_object_budget=1000)
    assert len(objects) <= 50


# ─── Phase 4 redesign tests: x-monotone, diversity, role separation ───

def _multi_sections(n: int = 4, width: float = 1000.0) -> list[SectionPlan]:
    return [
        SectionPlan(
            start_time=i * 5.0, end_time=(i + 1) * 5.0,
            start_x=i * width, end_x=(i + 1) * width,
            section_type="verse", gameplay_mode="cube",
            speed_state=SpeedState.NORMAL,
            density_target=0.7, decoration_intensity=0.8,
            trigger_intensity=0.4, difficulty_target=0.5,
        )
        for i in range(n)
    ]


def test_materialized_output_is_x_monotonic():
    """Sections should produce x-monotone output by construction."""
    config = MaterializationConfig(seed=1, decoration_density=1.0, target_object_count=500)
    objs = materialize_level_plans(_multi_sections(), config=config, total_object_budget=500)
    xs = [o.x for o in objs]
    violations = sum(1 for a, b in zip(xs, xs[1:]) if b < a)
    assert violations == 0, f"Expected x-monotone output, got {violations} violations"


def test_materialized_output_does_not_use_only_211():
    """The fallback ['211'] palette must be replaced with a diverse palette."""
    config = MaterializationConfig(seed=2, decoration_density=1.0, target_object_count=300)
    objs = materialize_level_plans(_multi_sections(), config=config, total_object_budget=300)
    if not objs:
        return
    unique = {o.object_id for o in objs}
    assert len(unique) > 1, f"Only one object_id used: {unique}"


def test_decoration_avoids_gameplay_y_band():
    """Decorations should not sit on the gameplay corridor (y < 180)."""
    config = MaterializationConfig(seed=3, decoration_density=1.0, target_object_count=400)
    objs = materialize_level_plans(_multi_sections(), config=config, total_object_budget=400)
    deco = [o for o in objs if o.role in {"fill_decoration", "background_detail"}]
    if deco:
        assert min(o.y for o in deco) >= 180, (
            f"Decoration encroaches gameplay corridor (min y={min(o.y for o in deco)})"
        )
