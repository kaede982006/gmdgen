# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from gmdgen.gd.time_mapping import (
    SPEED_PIXELS_PER_SECOND,
    SPEED_PORTAL_IDS,
    SpeedObject,
    SpeedState,
    build_beat_x_map,
    build_and_sort_speed_objects,
    build_speed_segments,
    compare_with_time_x_fixtures,
    pos_for_time_like_gd,
    round_trip_error_report,
    sort_speed_objects,
    time_for_pos_like_gd,
    value_for_speed_mod,
)


def test_song_offset_changes_audio_time_to_x() -> None:
    x = pos_for_time_like_gd(1.0, [], start_speed="normal", song_offset=0.25)
    assert x == value_for_speed_mod("normal") * 0.75


def test_time_mapping_round_trip_no_portal() -> None:
    x = pos_for_time_like_gd(
        2.25,
        [],
        start_speed="normal",
        song_offset=0.25,
    )
    recovered = time_for_pos_like_gd(
        x,
        [],
        start_speed="normal",
        song_offset=0.25,
    )
    assert abs(recovered - 2.25) < 1e-6


def test_speed_portal_recomputes_later_beat_spacing() -> None:
    speed_objects = build_and_sort_speed_objects(
        [(2.0, SpeedState.FAST)],
        start_speed="normal",
        song_offset=0.0,
    )
    x_after_portal = pos_for_time_like_gd(
        3.0,
        speed_objects,
        start_speed="normal",
        song_offset=0.0,
    )
    fixed_normal_x = value_for_speed_mod("normal") * 3.0

    assert x_after_portal != fixed_normal_x
    assert x_after_portal == (
        value_for_speed_mod("normal") * 2.0
        + value_for_speed_mod("fast") * 1.0
    )


def test_time_for_pos_roundtrip_with_speed_portal() -> None:
    speed_objects = build_and_sort_speed_objects(
        [(1.5, "fast"), (3.0, "normal")],
        start_speed="normal",
        song_offset=0.1,
    )
    x = pos_for_time_like_gd(
        3.5,
        speed_objects,
        start_speed="normal",
        song_offset=0.1,
    )
    t = time_for_pos_like_gd(
        x,
        speed_objects,
        start_speed="normal",
        song_offset=0.1,
    )
    assert abs(t - 3.5) < 1e-6


def test_time_mapping_round_trip_with_single_speed_portal() -> None:
    speed_objects = build_and_sort_speed_objects(
        [(1.0, "fast")],
        start_speed="normal",
        song_offset=0.0,
    )
    beats = [0.5, 1.0, 1.5, 2.0, 2.5]
    beat_map = build_beat_x_map(beats, speed_objects, "normal", 0.0)
    report = round_trip_error_report(beats, beat_map, speed_objects, "normal", 0.0)
    assert report["max_error"] < 1e-6


def test_time_mapping_round_trip_with_multiple_speed_portals() -> None:
    speed_objects = build_and_sort_speed_objects(
        [(1.0, "fast"), (2.0, "slow"), (3.0, "fastest")],
        start_speed="normal",
        song_offset=0.0,
    )
    beats = [idx * 0.25 for idx in range(1, 20)]
    beat_map = build_beat_x_map(beats, speed_objects, "normal", 0.0)
    report = round_trip_error_report(beats, beat_map, speed_objects, "normal", 0.0)
    assert report["max_error"] < 1e-6


def test_time_mapping_respects_song_offset() -> None:
    speed_objects = build_and_sort_speed_objects(
        [(1.5, "fast")],
        start_speed="normal",
        song_offset=0.25,
    )
    x = pos_for_time_like_gd(2.25, speed_objects, "normal", 0.25)
    recovered = time_for_pos_like_gd(x, speed_objects, "normal", 0.25)
    assert abs(recovered - 2.25) < 1e-6
    assert x == (
        value_for_speed_mod("normal") * 1.25
        + value_for_speed_mod("fast") * 0.75
    )


def test_speed_objects_are_sorted() -> None:
    unsorted = [
        SpeedObject(time=3.0, x=300.0, speed_state=SpeedState.FAST),
        SpeedObject(time=1.0, x=100.0, speed_state=SpeedState.SLOW),
        SpeedObject(time=2.0, x=200.0, speed_state=SpeedState.FASTER),
    ]
    sorted_objects = sort_speed_objects(reversed(unsorted))
    assert [obj.x for obj in sorted_objects] == [100.0, 200.0, 300.0]


def test_build_beat_x_map_recomputes_after_portal_change() -> None:
    beats = [1.0, 2.0, 3.0]
    no_portal = build_beat_x_map(beats, [], "normal", 0.0)
    with_portal = build_beat_x_map(
        beats,
        build_and_sort_speed_objects([(1.5, "fast")], start_speed="normal"),
        "normal",
        0.0,
    )
    assert no_portal[2] != with_portal[2]


def test_all_speed_states_have_positive_px_per_second() -> None:
    assert set(SPEED_PIXELS_PER_SECOND) == set(SpeedState)
    assert set(SPEED_PORTAL_IDS.values()) == set(SpeedState)
    assert all(value_for_speed_mod(state) > 0 for state in SpeedState)


def test_build_speed_segments_integrates_multiple_portals() -> None:
    speed_objects = build_and_sort_speed_objects(
        [(1.0, "fast"), (2.0, "slow")],
        start_speed="normal",
        song_offset=0.0,
    )
    segments = build_speed_segments(speed_objects, "normal", 0.0, duration=3.0)
    assert [segment.speed_state for segment in segments] == [
        SpeedState.NORMAL,
        SpeedState.FAST,
        SpeedState.SLOW,
    ]
    assert segments[-1].end_x == pos_for_time_like_gd(3.0, speed_objects, "normal", 0.0)


def test_compare_with_time_x_fixtures() -> None:
    fixture = {
        "time": 2.0,
        "expected_x": value_for_speed_mod("normal") * 2.0,
        "start_speed": "normal",
        "song_offset": 0.0,
    }
    report = compare_with_time_x_fixtures([fixture], tolerance=1e-6)
    assert report["passed"] is True
    assert report["checked_count"] == 1
