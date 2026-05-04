# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from gmdgen.gd.plans import ObjectPlan, SectionPlan
from gmdgen.gd.time_mapping import SpeedState
from gmdgen.generate.playability import (
    build_mode_physics_profile,
    estimate_trajectory_envelope,
    validate_trajectory_playability,
)
from gmdgen.generate.repairer import prune_impossible_spacing


def _section(mode: str = "cube", speed: SpeedState = SpeedState.NORMAL) -> SectionPlan:
    return SectionPlan(
        start_time=0.0,
        end_time=4.0,
        start_x=0.0,
        end_x=600.0,
        section_type="normal",
        gameplay_mode=mode,
        speed_state=speed,
        density_target=0.5,
        decoration_intensity=0.5,
        trigger_intensity=0.5,
        difficulty_target=0.5,
    )


def _obj(object_id: str, x: float, y: float, role: str) -> ObjectPlan:
    return ObjectPlan(object_id=object_id, x=x, y=y, role=role)


def test_trajectory_profile_exists_for_all_modes() -> None:
    for mode in ["cube", "ship", "ball", "ufo", "wave", "robot", "spider"]:
        profile = build_mode_physics_profile(mode, SpeedState.NORMAL, "normal")
        assert profile.mode == mode
        assert profile.min_obstacle_spacing > 0


def test_cube_trajectory_flags_too_close_hazard() -> None:
    section = _section("cube")
    warnings = validate_trajectory_playability(
        [section],
        [],
        [_obj("8", 120, 90, "hazard"), _obj("1", 140, 90, "structure")],
        difficulty="normal",
    )
    assert any(w.warning_type in {"cube_jump_arc_hazard", "hazard_margin"} for w in warnings)


def test_ship_corridor_too_narrow_warning() -> None:
    section = _section("ship")
    envelope = estimate_trajectory_envelope(
        section,
        object_plans=[_obj("1", 20, 100, "structure"), _obj("1", 30, 108, "structure")],
        difficulty="normal",
    )
    assert any(w.warning_type == "ship_corridor_too_narrow" for w in envelope.warnings)


def test_wave_excessive_input_density_warning() -> None:
    section = _section("wave", SpeedState.FASTER)
    objects = [_obj("36", 50 + idx * 12, 120, "beat_orb") for idx in range(20)]
    warnings = validate_trajectory_playability([section], [], objects, difficulty="easy")
    assert any(w.warning_type in {"trajectory_density", "trajectory_input_density"} for w in warnings)


def test_portal_transition_requires_recovery_margin() -> None:
    section = _section("cube", SpeedState.FAST)
    objects = [
        _obj("202", 100, 150, "speed_portal"),
        _obj("8", 130, 90, "hazard"),
    ]
    warnings = validate_trajectory_playability([section], [], objects, difficulty="normal")
    assert any(w.warning_type == "portal_transition_recovery" for w in warnings)


def test_difficulty_allows_tighter_timing() -> None:
    easy = build_mode_physics_profile("cube", SpeedState.NORMAL, "easy")
    demon = build_mode_physics_profile("cube", SpeedState.NORMAL, "demon")
    assert demon.min_obstacle_spacing < easy.min_obstacle_spacing


def test_repairer_reduces_trajectory_warnings() -> None:
    raw = [
        "1,1,2,100,3,90",
        "1,1,2,104,3,90",
        "1,1,2,140,3,90",
    ]
    repaired, removed = prune_impossible_spacing(raw, min_gap=20)
    assert removed > 0
    assert len(repaired) < len(raw)
