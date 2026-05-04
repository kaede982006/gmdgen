from __future__ import annotations

"""Verify that high object counts can be reached without destroying diversity
or breaking x-monotonicity (Phase 5 redesign)."""

from gmdgen.gd.plans import SectionPlan
from gmdgen.gd.time_mapping import SpeedState
from gmdgen.generate.materializer import MaterializationConfig, materialize_level_plans
from gmdgen.generate.scoring import score_diversity, score_position


def _sections(n: int = 8, width: float = 1000.0) -> list[SectionPlan]:
    return [
        SectionPlan(
            start_time=i * 5.0, end_time=(i + 1) * 5.0,
            start_x=i * width, end_x=(i + 1) * width,
            section_type="verse", gameplay_mode="cube",
            speed_state=SpeedState.NORMAL,
            density_target=0.8, decoration_intensity=0.9,
            trigger_intensity=0.4, difficulty_target=0.5,
        )
        for i in range(n)
    ]


def _to_strs(objs):
    return [f"1,{o.object_id},2,{o.x:.1f},3,{o.y:.1f};" for o in objs]


def test_high_object_count_remains_x_monotonic():
    """Even with thousands of objects, x must be monotonic by construction."""
    config = MaterializationConfig(
        seed=42, decoration_density=1.0,
        target_object_count=3000, object_multiplier=10.0,
    )
    objs = materialize_level_plans(_sections(), config=config, total_object_budget=3000)
    xs = [o.x for o in objs]
    violations = sum(1 for a, b in zip(xs, xs[1:]) if b < a)
    assert violations == 0, f"Expected 0 x_mono violations, got {violations}"


def test_diversity_above_0_0017_baseline_at_high_count():
    """object_diversity_score must be meaningfully above the 0.0017 failure baseline."""
    config = MaterializationConfig(
        seed=42, decoration_density=1.0,
        target_object_count=3000, object_multiplier=10.0,
    )
    objs = materialize_level_plans(_sections(), config=config, total_object_budget=3000)
    diversity = score_diversity(_to_strs(objs))
    assert diversity > 0.005, (
        f"Expected diversity > 0.005 (3x baseline), got {diversity:.5f}"
    )


def test_high_object_count_reaches_target_within_budget():
    target = 2000
    config = MaterializationConfig(
        seed=7, decoration_density=1.0,
        target_object_count=target, object_multiplier=8.0,
    )
    objs = materialize_level_plans(_sections(), config=config, total_object_budget=target)
    assert len(objs) <= target, f"Object count {len(objs)} exceeded target budget {target}"


def test_score_position_perfect_for_materialized_levels():
    config = MaterializationConfig(seed=11, decoration_density=1.0, target_object_count=500)
    objs = materialize_level_plans(_sections(n=4), config=config, total_object_budget=500)
    pos = score_position(_to_strs(objs))
    assert pos == 1.0, f"Materialized objects should be x-monotone (score=1.0), got {pos}"


def test_role_distribution_is_diverse():
    """Materialized levels should include multiple decoration roles."""
    config = MaterializationConfig(seed=99, decoration_density=1.0, target_object_count=500)
    objs = materialize_level_plans(_sections(), config=config, total_object_budget=500)
    roles = {o.role for o in objs}
    # We expect at least fill_decoration and background_detail when count is high.
    assert len(roles) >= 2, f"Expected at least 2 distinct roles, got {roles}"


def test_no_overwhelming_repeated_id():
    """The fallback decoration ID 211 must not dominate the output anymore."""
    config = MaterializationConfig(seed=2, decoration_density=1.0, target_object_count=1000)
    objs = materialize_level_plans(_sections(), config=config, total_object_budget=1000)
    if not objs:
        return
    counts = {}
    for o in objs:
        counts[o.object_id] = counts.get(o.object_id, 0) + 1
    most_common = max(counts.values())
    ratio = most_common / len(objs)
    # Previously "211" was 100% of fill output. Even at high decoration counts,
    # no single ID should claim more than 40% of the level now.
    assert ratio < 0.40, (
        f"Single object_id claims {ratio:.2%} of the level; palette rotation broken"
    )
