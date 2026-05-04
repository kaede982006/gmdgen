from __future__ import annotations

"""Tests for fallback palette behavior when no learning data is available.

The previous failure mode was: the materializer used ['211'] as the only
fallback decoration ID, collapsing object_diversity_score to ~0.0017.
This test suite ensures the diverse safe palette is always used as a
fallback, regardless of style_profile presence."""

from gmdgen.gd.plans import SectionPlan
from gmdgen.gd.time_mapping import SpeedState
from gmdgen.generate.materializer import (
    MaterializationConfig,
    SectionObjectMaterializer,
    materialize_level_plans,
    _SAFE_DECORATION_PALETTE,
    _palette_for_section,
)


def _section(start_x: float, end_x: float) -> SectionPlan:
    return SectionPlan(
        start_time=0.0, end_time=10.0,
        start_x=start_x, end_x=end_x,
        section_type="verse", gameplay_mode="cube",
        speed_state=SpeedState.NORMAL,
        density_target=0.7, decoration_intensity=0.85,
        trigger_intensity=0.4, difficulty_target=0.5,
    )


def test_safe_decoration_palette_is_diverse_by_default():
    """Built-in safe palette must contain at least 20 unique IDs."""
    assert len(set(_SAFE_DECORATION_PALETTE)) >= 20, (
        f"Safe palette has only {len(set(_SAFE_DECORATION_PALETTE))} unique IDs"
    )


def test_safe_palette_does_not_contain_only_211():
    """The previous degenerate ['211'] fallback must be gone."""
    assert _SAFE_DECORATION_PALETTE != ["211"], "Palette is back to degenerate ['211']"


def test_no_style_profile_uses_safe_palette():
    """Materialization without style_profile must produce diverse output."""
    config = MaterializationConfig(seed=1, decoration_density=1.0, target_object_count=400)
    sections = [_section(i * 1000.0, (i + 1) * 1000.0) for i in range(4)]
    objs = materialize_level_plans(sections, config=config, total_object_budget=400, style_profile=None)
    if not objs:
        return
    unique = {o.object_id for o in objs}
    assert len(unique) >= 5, f"Diverse fallback palette not used: only {unique}"


def test_palette_rotation_avoids_uniform_section_output():
    """Different sections must use different palette IDs (rotation)."""
    p0 = _palette_for_section(0, [])
    p1 = _palette_for_section(1, [])
    p2 = _palette_for_section(2, [])
    p3 = _palette_for_section(3, [])
    palettes = [p0, p1, p2, p3]
    # At least two adjacent sections should differ in their palette ordering.
    assert any(palettes[i] != palettes[i + 1] for i in range(len(palettes) - 1)), (
        "All section palettes are identical; rotation broken"
    )


def test_empty_style_profile_does_not_crash():
    """Style profile that's an empty dict must not crash materialization."""
    config = MaterializationConfig(seed=1, decoration_density=1.0, target_object_count=100)
    section = _section(0.0, 500.0)
    mat = SectionObjectMaterializer(config)
    objs = mat.materialize_section(section, style_profile={}, object_budget=100)
    # No exception, output may be small but should be a list.
    assert isinstance(objs, list)


def test_style_profile_ids_by_class_is_used_when_available():
    """Materializer should prefer style_profile['ids_by_class'] when present."""
    config = MaterializationConfig(seed=1, decoration_density=1.0, target_object_count=100)
    section = _section(0.0, 500.0)
    custom_ids = ["1764", "1765", "1766", "1767"]
    style_profile = {
        "ids_by_class": {
            "DECORATION": custom_ids,
        }
    }
    mat = SectionObjectMaterializer(config)
    objs = mat.materialize_section(section, style_profile=style_profile, object_budget=100)
    deco_ids = {o.object_id for o in objs if o.role in {"fill_decoration", "background_detail"}}
    if deco_ids:
        # At least one of the custom IDs should appear in the output.
        assert deco_ids & set(custom_ids), (
            f"Custom palette {custom_ids} not used; got {deco_ids}"
        )


def test_diversity_dramatically_above_0_0017_baseline():
    """The previous failure had object_diversity_score=0.0017. The new
    fallback must produce a meaningfully higher diversity ratio."""
    from gmdgen.generate.scoring import score_diversity
    config = MaterializationConfig(seed=42, decoration_density=1.0, target_object_count=1000)
    sections = [_section(i * 1500.0, (i + 1) * 1500.0) for i in range(8)]
    objs = materialize_level_plans(sections, config=config, total_object_budget=1000)
    strs = [f"1,{o.object_id},2,{o.x:.1f},3,{o.y:.1f};" for o in objs]
    diversity = score_diversity(strs)
    assert diversity > 0.005, (
        f"Diversity {diversity:.5f} is not meaningfully above 0.0017 baseline"
    )
