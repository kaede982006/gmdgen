# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from gmdgen.gd.plans import ObjectPlan, SectionPlan
from gmdgen.gd.time_mapping import SpeedState
from gmdgen.generate.repairer import prune_impossible_spacing
from gmdgen.generate.validator import validate_playability_v1


def _section(mode: str = "cube", speed: SpeedState = SpeedState.NORMAL) -> SectionPlan:
    return SectionPlan(
        start_time=0.0,
        end_time=10.0,
        start_x=0.0,
        end_x=2000.0,
        section_type="normal",
        gameplay_mode=mode,
        speed_state=speed,
        density_target=0.5,
        decoration_intensity=0.5,
        trigger_intensity=0.5,
        difficulty_target=0.5,
    )


def _obj(x: float, role: str = "structure", object_id: str = "1") -> ObjectPlan:
    return ObjectPlan(object_id=object_id, x=x, y=90.0, role=role)


def test_cube_min_spacing_violation() -> None:
    warnings = validate_playability_v1(
        object_plans=[_obj(100), _obj(110)],
        section_plans=[_section("cube")],
        difficulty="easy",
    )
    assert any(w.warning_type == "min_event_spacing" for w in warnings)


def test_wave_tighter_spacing_allowed_than_cube_or_configured() -> None:
    cube_warnings = validate_playability_v1(
        object_plans=[_obj(100), _obj(138)],
        section_plans=[_section("cube")],
        difficulty="easy",
    )
    wave_warnings = validate_playability_v1(
        object_plans=[_obj(100), _obj(138)],
        section_plans=[_section("wave")],
        difficulty="demon",
    )
    assert len(wave_warnings) <= len(cube_warnings)


def test_portal_requires_safety_margin() -> None:
    warnings = validate_playability_v1(
        object_plans=[
            _obj(100, role="speed_portal", object_id="202"),
            _obj(130, role="structure", object_id="1"),
        ],
        section_plans=[_section("cube", SpeedState.FAST)],
        difficulty="normal",
    )
    assert any(w.warning_type == "portal_safety_margin" for w in warnings)


def test_excessive_input_density_warning() -> None:
    warnings = validate_playability_v1(
        object_plans=[_obj(100 + idx * 20, role="beat_orb", object_id="36") for idx in range(8)],
        section_plans=[_section("cube")],
        difficulty="easy",
    )
    assert any(w.warning_type == "excessive_input_density" for w in warnings)


def test_difficulty_changes_threshold() -> None:
    easy = validate_playability_v1(
        object_plans=[_obj(100), _obj(135)],
        section_plans=[_section("cube")],
        difficulty="easy",
    )
    demon = validate_playability_v1(
        object_plans=[_obj(100), _obj(135)],
        section_plans=[_section("cube")],
        difficulty="demon",
    )
    assert len(demon) <= len(easy)


def test_repairer_prunes_impossible_spacing() -> None:
    objects = [
        "1,1,2,100,3,90",
        "1,1,2,104,3,90",
        "1,1,2,140,3,90",
    ]
    repaired, removed = prune_impossible_spacing(objects, min_gap=12)
    assert removed == 1
    assert len(repaired) == 2
