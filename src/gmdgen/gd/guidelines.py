from __future__ import annotations

from gmdgen.gd.time_mapping import (
    SpeedObject,
    SpeedState,
    pos_for_time_like_gd,
    speed_state_at_time,
    sync_error_for_x,
)


def build_guideline_string(beat_times: list[float], *, max_entries: int = 2048) -> str:
    """Build a compact editor-style beat guide string.

    GD guideline strings are version-sensitive, so this keeps a conservative
    time list that can be stored in LevelSettingsPlan and used by our own
    preview/round-trip checks. The final save encoder can later map it to the
    exact setting key after the project chooses a supported GD version map.
    """

    entries = [f"{idx}:{round(time_value, 5)}" for idx, time_value in enumerate(beat_times[:max_entries])]
    return "|".join(entries)


def beat_time_events(
    beat_times: list[float],
    *,
    speed_objects: list[SpeedObject],
    start_speed: SpeedState | str,
    song_offset: float,
    beat_snap_tolerance: float,
) -> list[dict[str, float | str]]:
    events: list[dict[str, float | str]] = []
    for beat_time in beat_times:
        x_pos = pos_for_time_like_gd(
            beat_time,
            speed_objects,
            start_speed=start_speed,
            song_offset=song_offset,
        )
        sync_error = sync_error_for_x(
            x=x_pos,
            expected_audio_time=beat_time,
            speed_objects=speed_objects,
            start_speed=start_speed,
            song_offset=song_offset,
        )
        state = speed_state_at_time(
            beat_time,
            speed_objects,
            start_speed=start_speed,
            song_offset=song_offset,
        )
        events.append(
            {
                "time": beat_time,
                "x": x_pos,
                "speed_state": state.value,
                "sync_error": sync_error,
                "within_tolerance": abs(sync_error) <= beat_snap_tolerance,
            }
        )
    return events
