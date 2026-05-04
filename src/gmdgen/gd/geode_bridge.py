from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

from gmdgen.gd.time_mapping import (
    SpeedObject,
    SpeedState,
    pos_for_time_like_gd,
    time_for_pos_like_gd,
)


@dataclass(slots=True)
class GeodeValidationResult:
    available: bool = False
    valid: bool = False
    fatal_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    object_count: int = 0
    trigger_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GeodeParsedLevelSummary:
    available: bool = False
    object_count: int = 0
    trigger_count: int = 0
    speed_portal_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GeodeRoundTripResult:
    available: bool = False
    round_trip_ok: bool = False
    object_count_before: int = 0
    object_count_after: int = 0
    warnings: list[str] = field(default_factory=list)
    fatal_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GeodeTriggerReport:
    available: bool = False
    unsupported_trigger_count: int = 0
    malformed_trigger_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GeodeImportSafetyReport:
    available: bool = False
    safe: bool = False
    warnings: list[str] = field(default_factory=list)
    fatal_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GeodeParityReport:
    available: bool
    checked: bool
    average_abs_x_error: float = 0.0
    max_abs_x_error: float = 0.0
    average_abs_time_error: float = 0.0
    max_abs_time_error: float = 0.0
    sample_count: int = 0
    passed: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GeodeBridge(Protocol):
    def is_available(self) -> bool: ...
    def get_version(self) -> str | None: ...
    def pos_for_time(
        self,
        time: float,
        speed_objects: Iterable[SpeedObject],
        start_speed: SpeedState | str,
        song_offset: float,
    ) -> float: ...
    def time_for_pos(
        self,
        x: float,
        speed_objects: Iterable[SpeedObject],
        start_speed: SpeedState | str,
        song_offset: float,
    ) -> float: ...
    def validate_level_string(self, level_string: str) -> GeodeValidationResult: ...
    def parse_level_string(self, level_string: str) -> GeodeParsedLevelSummary: ...
    def round_trip_level_string(self, level_string: str) -> GeodeRoundTripResult: ...
    def inspect_triggers(self, level_string: str) -> GeodeTriggerReport: ...
    def import_safety_check(self, level_string: str) -> GeodeImportSafetyReport: ...


class NullGeodeBridge:
    def is_available(self) -> bool:
        return False

    def get_version(self) -> str | None:
        return None

    def pos_for_time(
        self,
        time: float,
        speed_objects: Iterable[SpeedObject],
        start_speed: SpeedState | str,
        song_offset: float,
    ) -> float:
        raise RuntimeError("Geode bridge is unavailable")

    def time_for_pos(
        self,
        x: float,
        speed_objects: Iterable[SpeedObject],
        start_speed: SpeedState | str,
        song_offset: float,
    ) -> float:
        raise RuntimeError("Geode bridge is unavailable")

    def validate_level_string(self, level_string: str) -> GeodeValidationResult:
        return GeodeValidationResult(warnings=["geode_unavailable"])

    def parse_level_string(self, level_string: str) -> GeodeParsedLevelSummary:
        return GeodeParsedLevelSummary(warnings=["geode_unavailable"])

    def round_trip_level_string(self, level_string: str) -> GeodeRoundTripResult:
        return GeodeRoundTripResult(warnings=["geode_unavailable"])

    def inspect_triggers(self, level_string: str) -> GeodeTriggerReport:
        return GeodeTriggerReport(warnings=["geode_unavailable"])

    def import_safety_check(self, level_string: str) -> GeodeImportSafetyReport:
        return GeodeImportSafetyReport(warnings=["geode_unavailable"])


class OptionalGeodeFixtureBridge:
    """Fixture-backed bridge for parity tests.

    This does not claim runtime GD/Geode parity. It lets recorded Geode outputs
    be injected later and compared against the approximate Python mapper.
    """

    def __init__(self, fixture: dict[str, Any]) -> None:
        self.fixture = dict(fixture)
        self._samples = list(self.fixture.get("samples", []))

    def is_available(self) -> bool:
        return True

    def get_version(self) -> str | None:
        return str(self.fixture.get("geode_version") or self.fixture.get("version") or "fixture")

    def pos_for_time(
        self,
        time: float,
        speed_objects: Iterable[SpeedObject],
        start_speed: SpeedState | str,
        song_offset: float,
    ) -> float:
        sample = _nearest_sample(self._samples, "time", time)
        if sample and "expected_x" in sample:
            return float(sample["expected_x"])
        return pos_for_time_like_gd(time, speed_objects, start_speed, song_offset)

    def time_for_pos(
        self,
        x: float,
        speed_objects: Iterable[SpeedObject],
        start_speed: SpeedState | str,
        song_offset: float,
    ) -> float:
        sample = _nearest_sample(self._samples, "x", x)
        if sample and "expected_time" in sample:
            return float(sample["expected_time"])
        return time_for_pos_like_gd(x, speed_objects, start_speed, song_offset)

    def validate_level_string(self, level_string: str) -> GeodeValidationResult:
        return GeodeValidationResult(available=True, valid=True)

    def parse_level_string(self, level_string: str) -> GeodeParsedLevelSummary:
        objects = [part for part in level_string.split(";") if part.strip()]
        return GeodeParsedLevelSummary(available=True, object_count=len(objects))

    def round_trip_level_string(self, level_string: str) -> GeodeRoundTripResult:
        count = len([part for part in level_string.split(";") if part.strip()])
        return GeodeRoundTripResult(
            available=True,
            round_trip_ok=True,
            object_count_before=count,
            object_count_after=count,
        )

    def inspect_triggers(self, level_string: str) -> GeodeTriggerReport:
        return GeodeTriggerReport(available=True)

    def import_safety_check(self, level_string: str) -> GeodeImportSafetyReport:
        return GeodeImportSafetyReport(available=True, safe=True)


class ExternalProcessGeodeBridge:
    """JSON IPC skeleton for a future Geode helper executable."""

    def __init__(self, helper_path: str | Path, *, timeout_seconds: float = 5.0) -> None:
        self.helper_path = Path(helper_path)
        self.timeout_seconds = float(timeout_seconds)

    def is_available(self) -> bool:
        return self.helper_path.exists() and self.helper_path.is_file()

    def get_version(self) -> str | None:
        try:
            payload = self._request({"method": "version"})
        except Exception:
            return None
        return str(payload.get("version", "")) or None

    def pos_for_time(
        self,
        time: float,
        speed_objects: Iterable[SpeedObject],
        start_speed: SpeedState | str,
        song_offset: float,
    ) -> float:
        payload = self._request(
            {
                "method": "pos_for_time",
                "time": time,
                "speed_objects": [_speed_object_to_dict(obj) for obj in speed_objects],
                "start_speed": str(start_speed),
                "song_offset": song_offset,
            }
        )
        return float(payload["x"])

    def time_for_pos(
        self,
        x: float,
        speed_objects: Iterable[SpeedObject],
        start_speed: SpeedState | str,
        song_offset: float,
    ) -> float:
        payload = self._request(
            {
                "method": "time_for_pos",
                "x": x,
                "speed_objects": [_speed_object_to_dict(obj) for obj in speed_objects],
                "start_speed": str(start_speed),
                "song_offset": song_offset,
            }
        )
        return float(payload["time"])

    def validate_level_string(self, level_string: str) -> GeodeValidationResult:
        payload = self._request({"method": "validate_level_string", "level_string": level_string})
        return GeodeValidationResult(**payload)

    def parse_level_string(self, level_string: str) -> GeodeParsedLevelSummary:
        payload = self._request({"method": "parse_level_string", "level_string": level_string})
        return GeodeParsedLevelSummary(**payload)

    def round_trip_level_string(self, level_string: str) -> GeodeRoundTripResult:
        payload = self._request({"method": "round_trip_level_string", "level_string": level_string})
        return GeodeRoundTripResult(**payload)

    def inspect_triggers(self, level_string: str) -> GeodeTriggerReport:
        payload = self._request({"method": "inspect_triggers", "level_string": level_string})
        return GeodeTriggerReport(**payload)

    def import_safety_check(self, level_string: str) -> GeodeImportSafetyReport:
        payload = self._request({"method": "import_safety_check", "level_string": level_string})
        return GeodeImportSafetyReport(**payload)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.is_available():
            raise RuntimeError(f"Geode helper unavailable: {self.helper_path}")
        completed = subprocess.run(
            [str(self.helper_path)],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Geode helper failed: {completed.stderr.strip()}")
        result = json.loads(completed.stdout or "{}")
        if not isinstance(result, dict):
            raise RuntimeError("Geode helper returned non-object JSON")
        return result


def compare_time_mapping_with_geode(
    geode_bridge: GeodeBridge,
    beats: Iterable[Any],
    speed_objects: Iterable[SpeedObject],
    start_speed: SpeedState | str,
    song_offset: float,
    *,
    tolerance: float = 0.05,
) -> GeodeParityReport:
    if not geode_bridge.is_available():
        return GeodeParityReport(
            available=False,
            checked=False,
            warnings=["geode_unavailable_using_python_approximation"],
        )
    x_errors: list[float] = []
    time_errors: list[float] = []
    for beat in beats:
        beat_time = float(getattr(beat, "time", beat))
        python_x = pos_for_time_like_gd(beat_time, speed_objects, start_speed, song_offset)
        geode_x = geode_bridge.pos_for_time(beat_time, speed_objects, start_speed, song_offset)
        x_errors.append(abs(python_x - geode_x))
        python_time = time_for_pos_like_gd(geode_x, speed_objects, start_speed, song_offset)
        geode_time = geode_bridge.time_for_pos(geode_x, speed_objects, start_speed, song_offset)
        time_errors.append(abs(python_time - geode_time))
    max_x = max(x_errors, default=0.0)
    max_time = max(time_errors, default=0.0)
    return GeodeParityReport(
        available=True,
        checked=True,
        average_abs_x_error=sum(x_errors) / len(x_errors) if x_errors else 0.0,
        max_abs_x_error=max_x,
        average_abs_time_error=sum(time_errors) / len(time_errors) if time_errors else 0.0,
        max_abs_time_error=max_time,
        sample_count=len(x_errors),
        passed=max_x <= tolerance and max_time <= tolerance,
        warnings=[] if max_x <= tolerance and max_time <= tolerance else ["geode_time_x_parity_mismatch"],
    )


def load_geode_fixture_bridge(path: str | Path) -> OptionalGeodeFixtureBridge:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Geode fixture must be a JSON object")
    return OptionalGeodeFixtureBridge(payload)


def _nearest_sample(samples: list[dict[str, Any]], key: str, value: float) -> dict[str, Any] | None:
    if not samples:
        return None
    return min(samples, key=lambda sample: abs(float(sample.get(key, value)) - value))


def _speed_object_to_dict(obj: SpeedObject) -> dict[str, Any]:
    return {
        "time": obj.time,
        "x": obj.x,
        "speed_state": obj.speed_state.value,
        "object_id": obj.object_id,
    }
