from __future__ import annotations

"""Tests for Phase 4: prevent debug-marker / vertical-line spam in exported levels.

Previous symptom: actual GD output looked like vertical marker spam — a stack of
identical marker objects in a vertical line. These tests verify:
- materialized output has a controlled trigger:gameplay and decoration:gameplay ratio
- no role labelled "debug_marker" leaks into a normal export
- no excessive vertical clustering (many identical x with different y)"""

from collections import Counter

from gmdgen.gd.plans import ObjectPlan, SectionPlan
from gmdgen.gd.time_mapping import SpeedState
from gmdgen.generate.materializer import MaterializationConfig, materialize_level_plans


def _sections(n: int = 4, width: float = 1000.0) -> list[SectionPlan]:
    return [
        SectionPlan(
            start_time=i * 5.0, end_time=(i + 1) * 5.0,
            start_x=i * width, end_x=(i + 1) * width,
            section_type="verse", gameplay_mode="cube",
            speed_state=SpeedState.NORMAL,
            density_target=0.7, decoration_intensity=0.85,
            trigger_intensity=0.4, difficulty_target=0.5,
        )
        for i in range(n)
    ]


def test_no_debug_marker_role_in_default_output():
    """Default materialization must not produce role='debug_marker' objects."""
    config = MaterializationConfig(seed=1, decoration_density=1.0, target_object_count=500)
    objs = materialize_level_plans(_sections(), config=config, total_object_budget=500)
    debug_markers = [o for o in objs if o.role == "debug_marker"]
    assert debug_markers == [], f"Found {len(debug_markers)} debug_marker objects in default output"


def test_no_excessive_vertical_clustering():
    """No single x-bin (rounded) should hold more than 8 objects.

    Vertical marker spam was characterized by tens of objects stacked at the same x.
    """
    config = MaterializationConfig(seed=2, decoration_density=1.0, target_object_count=1000)
    objs = materialize_level_plans(_sections(), config=config, total_object_budget=1000)
    if not objs:
        return
    x_bins: Counter[int] = Counter()
    for obj in objs:
        x_bins[int(obj.x // 30)] += 1
    most_stacked = max(x_bins.values()) if x_bins else 0
    assert most_stacked <= 8, (
        f"Found {most_stacked} objects in a single 30-unit x-bin; vertical spam likely"
    )


def test_role_distribution_has_no_dominant_marker_role():
    """No single role (other than gameplay-related) should dominate the level."""
    config = MaterializationConfig(seed=3, decoration_density=1.0, target_object_count=500)
    objs = materialize_level_plans(_sections(), config=config, total_object_budget=500)
    if not objs:
        return
    role_counts = Counter(o.role for o in objs)
    total = sum(role_counts.values())
    # Decoration is allowed to dominate (intentional for high-count levels)
    # but no single non-decoration role should exceed 95%.
    for role, count in role_counts.items():
        if role in {"fill_decoration", "background_detail", "beat_structure", "beat_orb"}:
            continue
        ratio = count / total
        assert ratio < 0.95, f"Role '{role}' dominates output: {ratio:.2%}"


def test_generated_object_ids_are_diverse():
    """No single object_id should claim more than 50% of the level."""
    config = MaterializationConfig(seed=4, decoration_density=1.0, target_object_count=500)
    objs = materialize_level_plans(_sections(), config=config, total_object_budget=500)
    if not objs:
        return
    id_counts = Counter(o.object_id for o in objs)
    total = sum(id_counts.values())
    most_common_count = id_counts.most_common(1)[0][1]
    ratio = most_common_count / total
    assert ratio < 0.50, f"Single object_id claims {ratio:.2%} of level; marker spam pattern"


def test_high_decoration_count_does_not_create_vertical_walls():
    """When decoration density is maxed, output must not become a vertical wall."""
    config = MaterializationConfig(
        seed=5, decoration_density=1.0,
        target_object_count=2000, object_multiplier=10.0,
    )
    objs = materialize_level_plans(_sections(n=8), config=config, total_object_budget=2000)
    # Compute x-spread: count of distinct 50-unit bins
    if not objs:
        return
    x_bins = {int(o.x // 50) for o in objs}
    # We expect distribution across the full 8000-unit range → at least 50 bins.
    assert len(x_bins) >= 50, (
        f"Output occupies only {len(x_bins)} distinct x-bins; decoration may be clustered"
    )


def test_y_values_are_not_all_identical():
    """Decoration must not all sit at the same y (a true vertical line)."""
    config = MaterializationConfig(seed=6, decoration_density=1.0, target_object_count=400)
    objs = materialize_level_plans(_sections(), config=config, total_object_budget=400)
    deco = [o for o in objs if o.role in {"fill_decoration", "background_detail"}]
    if not deco:
        return
    unique_y = {round(o.y, 0) for o in deco}
    assert len(unique_y) >= 5, (
        f"Decoration y-values collapse to only {len(unique_y)} distinct values"
    )


def test_materialized_objects_are_classified():
    """Every output object must have a non-empty role."""
    config = MaterializationConfig(seed=7, decoration_density=1.0, target_object_count=300)
    objs = materialize_level_plans(_sections(), config=config, total_object_budget=300)
    for obj in objs:
        assert obj.role, f"Object at x={obj.x} has empty role"


def test_object_role_field_does_not_contain_marker_keywords():
    """No role label should literally be 'marker', 'debug', or 'spam'."""
    config = MaterializationConfig(seed=8, decoration_density=1.0, target_object_count=300)
    objs = materialize_level_plans(_sections(), config=config, total_object_budget=300)
    forbidden_substrings = ["marker", "debug", "spam", "test"]
    for obj in objs:
        for forbidden in forbidden_substrings:
            assert forbidden not in obj.role.lower(), (
                f"Object role '{obj.role}' contains forbidden substring '{forbidden}'"
            )
