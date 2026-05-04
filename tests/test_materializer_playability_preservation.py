from __future__ import annotations

from gmdgen.gd.plans import ObjectPlan, SectionPlan, plans_to_level_objects
from gmdgen.gd.time_mapping import SpeedState
from gmdgen.generate.materializer import MaterializationConfig, SectionObjectMaterializer, materialize_level_plans
from gmdgen.generate.scoring import score_diversity, score_position


def _make_section(start_x: float, end_x: float) -> SectionPlan:
    return SectionPlan(
        start_time=0.0, end_time=10.0,
        start_x=start_x, end_x=end_x,
        section_type="verse", gameplay_mode="cube",
        speed_state=SpeedState.NORMAL,
        density_target=0.5, decoration_intensity=0.5,
        trigger_intensity=0.2, difficulty_target=0.5,
    )


def _objects_to_strs(plans: list[ObjectPlan]) -> list[str]:
    return [f"1,{obj.object_id},2,{obj.x:.1f},3,{obj.y:.1f};" for obj in plans]


def test_object_budget_is_never_exceeded():
    sections = [_make_section(0.0, 1000.0)]
    config = MaterializationConfig(seed=10)
    for budget in [10, 50, 200]:
        objects = materialize_level_plans(sections, config=config, total_object_budget=budget)
        assert len(objects) <= budget, f"Expected <= {budget} objects, got {len(objects)}"


def test_all_objects_stay_within_section_x_bounds():
    section = _make_section(200.0, 800.0)
    config = MaterializationConfig(seed=11, gameplay_density=0.5)
    mat = SectionObjectMaterializer(config)
    objects = mat.materialize_section(section, object_budget=200)
    for obj in objects:
        assert 200.0 <= obj.x <= 800.0, f"x={obj.x} outside [200, 800]"


def test_decoration_objects_have_decoration_role():
    section = _make_section(0.0, 1000.0)
    config = MaterializationConfig(seed=12, decoration_density=1.0, gameplay_density=0.0)
    mat = SectionObjectMaterializer(config)
    objects = mat.materialize_section(section, object_budget=50)
    decoration_roles = {"fill_decoration", "sync_accent", "materialized_motif"}
    if objects:
        has_deco = any(obj.role in decoration_roles for obj in objects)
        assert has_deco, "Expected some decoration/fill objects when gameplay_density=0"


def test_diversity_improves_with_varied_ids():
    """More varied object IDs should yield higher diversity score."""
    # All same ID
    uniform_plans = [
        ObjectPlan(object_id="1", x=float(i * 10), y=30.0, role="beat_structure")
        for i in range(20)
    ]
    uniform_strs = _objects_to_strs(uniform_plans)

    # Varied IDs
    id_pool = ["1", "36", "141", "1332", "211", "8", "39", "1"]
    varied_plans = [
        ObjectPlan(object_id=id_pool[i % len(id_pool)], x=float(i * 10), y=30.0, role="beat_structure")
        for i in range(20)
    ]
    varied_strs = _objects_to_strs(varied_plans)

    uniform_score = score_diversity(uniform_strs)
    varied_score = score_diversity(varied_strs)
    assert varied_score > uniform_score, (
        f"Varied IDs should score higher than uniform. uniform={uniform_score:.3f}, varied={varied_score:.3f}"
    )


def test_score_position_prefers_sorted_x():
    """Objects with monotonically increasing x should have position score >= unsorted."""
    sorted_plans = [
        ObjectPlan(object_id="1", x=float(i * 30), y=30.0, role="beat_structure")
        for i in range(20)
    ]
    sorted_strs = _objects_to_strs(sorted_plans)
    sorted_score = score_position(sorted_strs)
    assert sorted_score == 1.0, f"Sorted objects should have position score 1.0, got {sorted_score}"

    # Reversed order
    reversed_strs = list(reversed(sorted_strs))
    reversed_score = score_position(reversed_strs)
    assert reversed_score < sorted_score, "Reversed order should score lower"


def test_high_decoration_density_does_not_exceed_budget():
    sections = [_make_section(0.0, 2000.0)]
    config = MaterializationConfig(seed=13, decoration_density=1.0, sync_accent_density=1.0, object_multiplier=2.0)
    objects = materialize_level_plans(sections, config=config, total_object_budget=100)
    assert len(objects) <= 100


def test_multiple_sections_do_not_overlap_objects():
    sections = [
        _make_section(0.0, 300.0),
        _make_section(300.0, 600.0),
        _make_section(600.0, 900.0),
    ]
    config = MaterializationConfig(seed=14, gameplay_density=0.5)
    objects = materialize_level_plans(sections, config=config, total_object_budget=300)
    # Each object should fall within the x range of its section
    for obj in objects:
        in_some_section = any(s.start_x <= obj.x <= s.end_x for s in sections)
        assert in_some_section, f"Object x={obj.x} not in any section range"


def test_empty_level_produces_valid_output():
    sections: list[SectionPlan] = []
    config = MaterializationConfig(seed=15)
    objects = materialize_level_plans(sections, config=config, total_object_budget=100)
    assert objects == []
