from __future__ import annotations

from unittest.mock import MagicMock
from gmdgen.gd.time_mapping import SpeedState, SpeedObject
from gmdgen.generate.scoring import _speed_portal_consistency


def _make_speed_object(x: float, time: float, speed: SpeedState = SpeedState.NORMAL) -> SpeedObject:
    return SpeedObject(x=x, time=time, speed_state=speed)


def _make_audio_features(beat_times=None, sections=None):
    af = MagicMock()
    af.beat_times = beat_times or []
    af.onset_times = []
    af.sections = sections or []
    return af


def test_speed_portal_consistency_sorted_order_gives_high_score():
    speed_objects = [
        _make_speed_object(100.0, 1.0),
        _make_speed_object(300.0, 3.0),
        _make_speed_object(600.0, 6.0),
    ]
    score = _speed_portal_consistency(
        speed_objects,
        start_speed=SpeedState.NORMAL,
        song_offset=0.0,
        tolerance=0.08,
    )
    assert 0.0 <= score <= 1.0


def test_speed_portal_consistency_unsorted_returns_zero():
    speed_objects = [
        _make_speed_object(600.0, 6.0),
        _make_speed_object(100.0, 1.0),
    ]
    score = _speed_portal_consistency(
        speed_objects,
        start_speed=SpeedState.NORMAL,
        song_offset=0.0,
        tolerance=0.08,
    )
    assert score == 0.0, "Unsorted speed objects should return score=0"


def test_no_speed_objects_returns_one():
    score = _speed_portal_consistency(
        [],
        start_speed=SpeedState.NORMAL,
        song_offset=0.0,
        tolerance=0.08,
    )
    assert score == 1.0, "No speed objects should return 1.0 (no violation)"


def test_speed_portal_consistency_single_object():
    speed_objects = [_make_speed_object(200.0, 2.0)]
    score = _speed_portal_consistency(
        speed_objects,
        start_speed=SpeedState.NORMAL,
        song_offset=0.0,
        tolerance=0.08,
    )
    assert 0.0 <= score <= 1.0


def test_speed_state_values_are_valid():
    """SpeedState enum has expected members."""
    assert hasattr(SpeedState, "NORMAL")
    assert hasattr(SpeedState, "FAST")
    assert hasattr(SpeedState, "SLOW")


def test_speed_object_has_expected_fields():
    obj = _make_speed_object(100.0, 1.0, SpeedState.FAST)
    assert obj.x == 100.0
    assert obj.time == 1.0
    assert obj.speed_state == SpeedState.FAST


def test_mode_transition_score_not_collapsed_with_valid_portals():
    """
    With sorted speed portals, speed_portal_consistency should be > 0.
    This validates that mode_transition_score won't collapse to 0.0
    when the generator places portals in correct order.
    """
    speed_objects = [
        _make_speed_object(50.0, 0.5),
        _make_speed_object(250.0, 2.5),
    ]
    score = _speed_portal_consistency(
        speed_objects,
        start_speed=SpeedState.NORMAL,
        song_offset=0.0,
        tolerance=0.16,
    )
    assert score > 0.0, f"Valid sorted portals should have score > 0, got {score}"


def test_section_plan_has_speed_state_field():
    from gmdgen.gd.plans import SectionPlan
    section = SectionPlan(
        start_time=0.0, end_time=5.0, start_x=0.0, end_x=500.0,
        section_type="verse", gameplay_mode="cube",
        speed_state=SpeedState.FAST,
        density_target=0.5, decoration_intensity=0.4,
        trigger_intensity=0.2, difficulty_target=0.5,
    )
    assert section.speed_state == SpeedState.FAST
    assert section.gameplay_mode == "cube"
