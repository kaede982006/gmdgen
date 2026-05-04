from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


class SpeedState(str, Enum):
    SLOW = "slow"
    NORMAL = "normal"
    FAST = "fast"
    FASTER = "faster"
    FASTEST = "fastest"


# Approximate mapper note:
# These are practical GD horizontal speeds in pixels/second. The decompiled
# LevelTools.cpp in base only exposes the signatures for posForTime/timeForPos,
# so this module keeps the same piecewise integration shape with known GD speed
# states instead of pretending raw BPM can determine X distance. Fixture parity
# against real GD/Geode LevelTools output should replace these constants once
# those runtime samples are available.
SPEED_PIXELS_PER_SECOND: dict[SpeedState, float] = {
    SpeedState.SLOW: 251.16,
    SpeedState.NORMAL: 311.58,
    SpeedState.FAST: 387.42,
    SpeedState.FASTER: 468.0,
    SpeedState.FASTEST: 576.0,
}

SPEED_PORTAL_IDS: dict[str, SpeedState] = {
    "200": SpeedState.SLOW,
    "201": SpeedState.NORMAL,
    "202": SpeedState.FAST,
    "203": SpeedState.FASTER,
    "1334": SpeedState.FASTEST,
}

SPEED_STATE_TO_PORTAL_ID: dict[SpeedState, str] = {
    state: object_id for object_id, state in SPEED_PORTAL_IDS.items()
}

_ALIASES: dict[str, SpeedState] = {
    "0.5x": SpeedState.SLOW,
    "half": SpeedState.SLOW,
    "slow": SpeedState.SLOW,
    "1x": SpeedState.NORMAL,
    "normal": SpeedState.NORMAL,
    "2x": SpeedState.FAST,
    "double": SpeedState.FAST,
    "fast": SpeedState.FAST,
    "3x": SpeedState.FASTER,
    "triple": SpeedState.FASTER,
    "faster": SpeedState.FASTER,
    "4x": SpeedState.FASTEST,
    "quadruple": SpeedState.FASTEST,
    "fastest": SpeedState.FASTEST,
}


@dataclass(frozen=True)
class SpeedObject:
    """A speed portal/object in audio-time coordinates.

    time is the audio event time. x is derived through pos_for_time_like_gd after
    applying song_offset and all earlier speed objects.
    """

    time: float
    x: float
    speed_state: SpeedState
    object_id: str | None = None
    source: str = "speed_portal"


@dataclass(frozen=True)
class SpeedSegment:
    """Integrated speed segment used by the approximate LevelTools mapper.

    start_time/end_time are audio timeline times. start_x/end_x are level X
    positions reached by integrating the active speed state over the segment.
    """

    start_time: float
    end_time: float
    start_x: float
    end_x: float
    speed_state: SpeedState
    px_per_second: float
    source_object_id: str | None = None
    source_x: float | None = None


def normalize_speed_state(value: SpeedState | str | None) -> SpeedState:
    if isinstance(value, SpeedState):
        return value
    if value is None:
        return SpeedState.NORMAL
    key = str(value).strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(f"Unknown GD speed state: {value!r}")


def value_for_speed_mod(speed_state: SpeedState | str | None) -> float:
    """Conceptual equivalent of LevelTools::valueForSpeedMod.

    The return value is a distance rate, not a BPM multiplier. This distinction
    is why beat spacing must be recomputed whenever speed state changes.
    """

    return SPEED_PIXELS_PER_SECOND[normalize_speed_state(speed_state)]


def speed_state_for_portal_id(object_id: str | None) -> SpeedState | None:
    if object_id is None:
        return None
    return SPEED_PORTAL_IDS.get(str(object_id))


def portal_id_for_speed_state(speed_state: SpeedState | str | None) -> str:
    return SPEED_STATE_TO_PORTAL_ID[normalize_speed_state(speed_state)]


def sort_speed_objects(speed_objects: Iterable[SpeedObject]) -> list[SpeedObject]:
    """Equivalent intent of LevelTools::sortSpeedObjects."""

    return sorted(speed_objects, key=lambda obj: (obj.x, obj.time, obj.object_id or ""))


def build_speed_segments(
    speed_objects: Iterable[SpeedObject] | None = None,
    start_speed: SpeedState | str | None = SpeedState.NORMAL,
    song_offset: float = 0.0,
    duration: float | None = None,
) -> list[SpeedSegment]:
    """Build integrated speed segments from sorted speed objects.

    This mirrors the shape of LevelTools::posForTimeInternal: active speed is
    piecewise constant, portals are applied in X order, and song_offset shifts
    audio time onto gameplay time before integration.
    """

    current_speed = normalize_speed_state(start_speed)
    last_game_time = 0.0
    last_audio_time = float(song_offset)
    current_x = 0.0
    segments: list[SpeedSegment] = []
    duration_game_time = (
        _gameplay_time(duration, song_offset) if duration is not None else None
    )

    for speed_object in sort_speed_objects(speed_objects or []):
        change_game_time = _gameplay_time(speed_object.time, song_offset)
        if duration_game_time is not None and change_game_time > duration_game_time:
            break
        if change_game_time <= last_game_time:
            current_speed = speed_object.speed_state
            continue

        px_per_second = value_for_speed_mod(current_speed)
        end_x = current_x + (change_game_time - last_game_time) * px_per_second
        segments.append(
            SpeedSegment(
                start_time=last_audio_time,
                end_time=change_game_time + song_offset,
                start_x=current_x,
                end_x=end_x,
                speed_state=current_speed,
                px_per_second=px_per_second,
                source_object_id=speed_object.object_id,
                source_x=speed_object.x,
            )
        )
        current_x = end_x
        last_game_time = change_game_time
        last_audio_time = change_game_time + song_offset
        current_speed = speed_object.speed_state

    if duration_game_time is not None and duration_game_time > last_game_time:
        px_per_second = value_for_speed_mod(current_speed)
        end_x = current_x + (duration_game_time - last_game_time) * px_per_second
        segments.append(
            SpeedSegment(
                start_time=last_audio_time,
                end_time=duration_game_time + song_offset,
                start_x=current_x,
                end_x=end_x,
                speed_state=current_speed,
                px_per_second=px_per_second,
            )
        )

    return segments


def _beat_time(beat: Any) -> float:
    if isinstance(beat, (int, float)):
        return float(beat)
    if isinstance(beat, dict):
        return float(beat.get("time", 0.0))
    return float(getattr(beat, "time", 0.0))


def _beat_index(index: int, beat: Any) -> int:
    if isinstance(beat, dict) and "index" in beat:
        return int(beat["index"])
    if not isinstance(beat, (int, float)) and hasattr(beat, "index"):
        return int(getattr(beat, "index"))
    return index


def _gameplay_time(audio_time: float, song_offset: float) -> float:
    return max(0.0, float(audio_time) - float(song_offset))


def _segment_changes(
    speed_objects: Iterable[SpeedObject],
    *,
    song_offset: float,
) -> list[SpeedObject]:
    return sorted(
        speed_objects,
        key=lambda obj: (_gameplay_time(obj.time, song_offset), obj.x),
    )


def pos_for_time_like_gd(
    audio_time: float,
    speed_objects: Iterable[SpeedObject] | None = None,
    start_speed: SpeedState | str | None = SpeedState.NORMAL,
    song_offset: float = 0.0,
) -> float:
    """Piecewise time-to-X mapping following GD LevelTools' conceptual shape.

    audio_time is measured on the song timeline. song_offset is applied first,
    then the path is integrated with the active speed state. This intentionally
    avoids fixed beat spacing.
    """

    target_time = _gameplay_time(audio_time, song_offset)
    if target_time <= 0.0:
        return 0.0

    segments = build_speed_segments(
        speed_objects,
        start_speed=start_speed,
        song_offset=song_offset,
        duration=audio_time,
    )
    return segments[-1].end_x if segments else 0.0


def time_for_pos_like_gd(
    x_pos: float,
    speed_objects: Iterable[SpeedObject] | None = None,
    start_speed: SpeedState | str | None = SpeedState.NORMAL,
    song_offset: float = 0.0,
) -> float:
    """Inverse mapping for pos_for_time_like_gd.

    Returns audio timeline time, so callers can compare it directly against
    beat/onset times and compute sync error.
    """

    target_x = max(0.0, float(x_pos))
    if target_x <= 0.0:
        return float(song_offset)

    sorted_objects = sort_speed_objects(speed_objects or [])
    segments = build_speed_segments(
        sorted_objects,
        start_speed=start_speed,
        song_offset=song_offset,
        duration=max((obj.time for obj in sorted_objects), default=song_offset),
    )
    for segment in segments:
        if target_x <= segment.end_x:
            dt = (target_x - segment.start_x) / max(segment.px_per_second, 1e-9)
            return segment.start_time + dt

    if segments:
        last_segment = segments[-1]
        current_speed = (
            sorted_objects[-1].speed_state if sorted_objects else last_segment.speed_state
        )
        current_x = last_segment.end_x
        last_time = last_segment.end_time
    else:
        current_speed = normalize_speed_state(start_speed)
        current_x = 0.0
        last_time = float(song_offset)

    dt = (target_x - current_x) / value_for_speed_mod(current_speed)
    return last_time + dt


def build_and_sort_speed_objects(
    candidates: Iterable[tuple[float, SpeedState | str]],
    *,
    start_speed: SpeedState | str | None = SpeedState.NORMAL,
    song_offset: float = 0.0,
) -> list[SpeedObject]:
    """Build speed objects from audio-time portal candidates.

    The x of each candidate is computed after all earlier candidates have been
    integrated. This is the critical recomputation missing from map-only
    generators.
    """

    built: list[SpeedObject] = []
    for time_value, state_value in sorted(candidates, key=lambda item: item[0]):
        state = normalize_speed_state(state_value)
        x_pos = pos_for_time_like_gd(
            time_value,
            built,
            start_speed=start_speed,
            song_offset=song_offset,
        )
        built.append(
            SpeedObject(
                time=float(time_value),
                x=x_pos,
                speed_state=state,
                object_id=portal_id_for_speed_state(state),
            )
        )
    return sort_speed_objects(built)


def build_beat_x_map(
    beats: Iterable[Any],
    speed_objects: Iterable[SpeedObject] | None = None,
    start_speed: SpeedState | str | None = SpeedState.NORMAL,
    song_offset: float = 0.0,
) -> dict[int, float]:
    """Build a beat-index to X map after speed objects are sorted/recomputed."""

    sorted_speed_objects = sort_speed_objects(speed_objects or [])
    beat_x_map: dict[int, float] = {}
    for idx, beat in enumerate(beats):
        beat_x_map[_beat_index(idx, beat)] = pos_for_time_like_gd(
            _beat_time(beat),
            sorted_speed_objects,
            start_speed=start_speed,
            song_offset=song_offset,
        )
    return beat_x_map


def round_trip_error_report(
    beats: Iterable[Any],
    beat_x_map: dict[int, float],
    speed_objects: Iterable[SpeedObject] | None = None,
    start_speed: SpeedState | str | None = SpeedState.NORMAL,
    song_offset: float = 0.0,
) -> dict[str, float | int | list[dict[str, float | int]]]:
    """Measure beat time -> X -> recovered time error."""

    sorted_speed_objects = sort_speed_objects(speed_objects or [])
    errors: list[float] = []
    invalid_count = 0
    samples: list[dict[str, float | int]] = []
    for idx, beat in enumerate(beats):
        beat_idx = _beat_index(idx, beat)
        expected_time = _beat_time(beat)
        x_pos = beat_x_map.get(beat_idx)
        if x_pos is None:
            invalid_count += 1
            continue
        recovered_time = time_for_pos_like_gd(
            x_pos,
            sorted_speed_objects,
            start_speed=start_speed,
            song_offset=song_offset,
        )
        error = abs(recovered_time - expected_time)
        errors.append(error)
        if len(samples) < 32:
            samples.append(
                {
                    "beat_index": beat_idx,
                    "time": expected_time,
                    "x": x_pos,
                    "recovered_time": recovered_time,
                    "error": error,
                }
            )

    average_error = sum(errors) / len(errors) if errors else 0.0
    max_error = max(errors, default=0.0)
    return {
        "average_error": average_error,
        "max_error": max_error,
        "invalid_count": invalid_count,
        "checked_count": len(errors),
        "samples": samples,
    }


def compare_with_time_x_fixtures(
    fixtures: Iterable[dict[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    """Compare mapper output with fixture samples.

    Each fixture may contain:
    - time, expected_x
    - x, expected_time
    - start_speed, song_offset
    - speed_objects as SpeedObject instances or dicts with time/x/speed_state.
    """

    failures: list[dict[str, Any]] = []
    max_error = 0.0
    checked = 0
    for fixture in fixtures:
        start_speed = fixture.get("start_speed", SpeedState.NORMAL)
        song_offset = float(fixture.get("song_offset", 0.0))
        speed_objects = [
            item if isinstance(item, SpeedObject) else SpeedObject(
                time=float(item.get("time", 0.0)),
                x=float(item.get("x", 0.0)),
                speed_state=normalize_speed_state(item.get("speed_state", "normal")),
                object_id=item.get("object_id"),
            )
            for item in fixture.get("speed_objects", [])
        ]

        if "time" in fixture and "expected_x" in fixture:
            checked += 1
            actual_x = pos_for_time_like_gd(
                float(fixture["time"]),
                speed_objects,
                start_speed=start_speed,
                song_offset=song_offset,
            )
            error = abs(actual_x - float(fixture["expected_x"]))
            max_error = max(max_error, error)
            if error > tolerance:
                failures.append({"fixture": fixture, "actual_x": actual_x, "error": error})

        if "x" in fixture and "expected_time" in fixture:
            checked += 1
            actual_time = time_for_pos_like_gd(
                float(fixture["x"]),
                speed_objects,
                start_speed=start_speed,
                song_offset=song_offset,
            )
            error = abs(actual_time - float(fixture["expected_time"]))
            max_error = max(max_error, error)
            if error > tolerance:
                failures.append({"fixture": fixture, "actual_time": actual_time, "error": error})

    return {
        "passed": not failures,
        "checked_count": checked,
        "max_error": max_error,
        "failures": failures,
    }


def speed_state_at_time(
    audio_time: float,
    speed_objects: Iterable[SpeedObject] | None = None,
    start_speed: SpeedState | str | None = SpeedState.NORMAL,
    song_offset: float = 0.0,
) -> SpeedState:
    current = normalize_speed_state(start_speed)
    target = _gameplay_time(audio_time, song_offset)
    for speed_object in _segment_changes(speed_objects or [], song_offset=song_offset):
        if _gameplay_time(speed_object.time, song_offset) > target:
            break
        current = speed_object.speed_state
    return current


def sync_error_for_x(
    *,
    x: float,
    expected_audio_time: float,
    speed_objects: Iterable[SpeedObject] | None = None,
    start_speed: SpeedState | str | None = SpeedState.NORMAL,
    song_offset: float = 0.0,
) -> float:
    actual_time = time_for_pos_like_gd(
        x,
        speed_objects,
        start_speed=start_speed,
        song_offset=song_offset,
    )
    return actual_time - expected_audio_time
