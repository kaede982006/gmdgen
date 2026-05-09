# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from gmdgen.gd.time_mapping import (
    SpeedObject,
    SpeedState,
    normalize_speed_state,
    pos_for_time_like_gd,
    time_for_pos_like_gd,
)


@dataclass(slots=True)
class TimeXSample:
    time: float | None = None
    expected_x: float | None = None
    x: float | None = None
    expected_time: float | None = None
    tolerance: float | None = None
    source: str = "synthetic_approximate"
    notes: str = ""


@dataclass(slots=True)
class TimeXFixture:
    name: str
    start_speed: SpeedState = SpeedState.NORMAL
    song_offset: float = 0.0
    speed_objects: list[SpeedObject] = field(default_factory=list)
    samples: list[TimeXSample] = field(default_factory=list)
    source: str = "synthetic_approximate"
    notes: str = ""


@dataclass(slots=True)
class TimeXFixtureComparison:
    fixture_name: str
    sample_count: int
    average_abs_x_error: float
    max_abs_x_error: float
    average_abs_time_error: float
    max_abs_time_error: float
    failed_samples: list[dict]
    passed: bool
    tolerance: float

    def to_dict(self) -> dict:
        return {
            "fixture_name": self.fixture_name,
            "sample_count": self.sample_count,
            "average_abs_x_error": self.average_abs_x_error,
            "max_abs_x_error": self.max_abs_x_error,
            "average_abs_time_error": self.average_abs_time_error,
            "max_abs_time_error": self.max_abs_time_error,
            "failed_samples": list(self.failed_samples),
            "passed": self.passed,
            "tolerance": self.tolerance,
        }


def load_time_x_fixture(path: str | Path) -> TimeXFixture:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json_fixture(path)
    if suffix == ".csv":
        return _load_csv_fixture(path)
    raise ValueError(f"Unsupported time-X fixture format: {path.suffix}")


def load_time_x_fixtures(directory: str | Path) -> list[TimeXFixture]:
    directory = Path(directory)
    if not directory.exists():
        return []
    fixtures: list[TimeXFixture] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in {".json", ".csv"}:
            fixtures.append(load_time_x_fixture(path))
    return fixtures


def compare_with_time_x_fixtures(
    fixtures: Iterable[TimeXFixture],
    mapper: Callable[..., float] | None = None,
    tolerance: float | None = None,
) -> list[TimeXFixtureComparison]:
    # mapper is intentionally generic for future Geode parity harnesses. When
    # omitted, the current approximate mapper is used.
    mapper = mapper or pos_for_time_like_gd
    results: list[TimeXFixtureComparison] = []
    for fixture in fixtures:
        x_errors: list[float] = []
        time_errors: list[float] = []
        failed_samples: list[dict] = []
        default_tolerance = float(tolerance if tolerance is not None else 1e-5)
        checked = 0
        for index, sample in enumerate(fixture.samples):
            sample_tolerance = float(sample.tolerance if sample.tolerance is not None else default_tolerance)
            if sample.time is not None and sample.expected_x is not None:
                checked += 1
                actual_x = mapper(
                    sample.time,
                    fixture.speed_objects,
                    start_speed=fixture.start_speed,
                    song_offset=fixture.song_offset,
                )
                error = abs(actual_x - sample.expected_x)
                x_errors.append(error)
                if error > sample_tolerance:
                    failed_samples.append(
                        {
                            "sample_index": index,
                            "kind": "time_to_x",
                            "expected": sample.expected_x,
                            "actual": actual_x,
                            "error": error,
                            "tolerance": sample_tolerance,
                        }
                    )
            if sample.x is not None and sample.expected_time is not None:
                checked += 1
                actual_time = time_for_pos_like_gd(
                    sample.x,
                    fixture.speed_objects,
                    start_speed=fixture.start_speed,
                    song_offset=fixture.song_offset,
                )
                error = abs(actual_time - sample.expected_time)
                time_errors.append(error)
                if error > sample_tolerance:
                    failed_samples.append(
                        {
                            "sample_index": index,
                            "kind": "x_to_time",
                            "expected": sample.expected_time,
                            "actual": actual_time,
                            "error": error,
                            "tolerance": sample_tolerance,
                        }
                    )

        results.append(
            TimeXFixtureComparison(
                fixture_name=fixture.name,
                sample_count=checked,
                average_abs_x_error=_mean(x_errors),
                max_abs_x_error=max(x_errors, default=0.0),
                average_abs_time_error=_mean(time_errors),
                max_abs_time_error=max(time_errors, default=0.0),
                failed_samples=failed_samples,
                passed=not failed_samples,
                tolerance=default_tolerance,
            )
        )
    return results


def summarize_time_x_fixture_errors(results: Iterable[TimeXFixtureComparison]) -> dict:
    results = list(results)
    return {
        "fixture_count": len(results),
        "passed": all(result.passed for result in results),
        "sample_count": sum(result.sample_count for result in results),
        "average_abs_x_error": _mean([result.average_abs_x_error for result in results]),
        "max_abs_x_error": max((result.max_abs_x_error for result in results), default=0.0),
        "average_abs_time_error": _mean([result.average_abs_time_error for result in results]),
        "max_abs_time_error": max((result.max_abs_time_error for result in results), default=0.0),
        "failed_sample_count": sum(len(result.failed_samples) for result in results),
        "results": [result.to_dict() for result in results],
    }


def _load_json_fixture(path: Path) -> TimeXFixture:
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Future Geode exporters may wrap mapper samples under "fixture" or
    # "time_x_fixture". Accept both to keep the format easy to evolve.
    if "fixture" in payload and isinstance(payload["fixture"], dict):
        payload = payload["fixture"]
    if "time_x_fixture" in payload and isinstance(payload["time_x_fixture"], dict):
        payload = payload["time_x_fixture"]
    return _fixture_from_mapping(payload, fallback_name=path.stem)


def _load_csv_fixture(path: Path) -> TimeXFixture:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return TimeXFixture(name=path.stem)
    first = rows[0]
    speed_objects: list[SpeedObject] = []
    fixture = TimeXFixture(
        name=first.get("fixture_name") or path.stem,
        start_speed=normalize_speed_state(first.get("start_speed") or "normal"),
        song_offset=_float_or_default(first.get("song_offset"), 0.0),
        speed_objects=speed_objects,
        source=first.get("source") or "synthetic_approximate",
    )
    for row in rows:
        if row.get("speed_object_time") and row.get("speed_object_state"):
            speed_objects.append(
                SpeedObject(
                    time=_float_or_default(row.get("speed_object_time"), 0.0),
                    x=_float_or_default(row.get("speed_object_x"), 0.0),
                    speed_state=normalize_speed_state(row.get("speed_object_state")),
                    object_id=row.get("speed_object_id") or None,
                )
            )
        fixture.samples.append(
            TimeXSample(
                time=_optional_float(row.get("time")),
                expected_x=_optional_float(row.get("expected_x")),
                x=_optional_float(row.get("x")),
                expected_time=_optional_float(row.get("expected_time")),
                tolerance=_optional_float(row.get("tolerance")),
                source=row.get("source") or fixture.source,
                notes=row.get("notes") or "",
            )
        )
    return fixture


def _fixture_from_mapping(payload: dict, *, fallback_name: str) -> TimeXFixture:
    speed_objects = [
        SpeedObject(
            time=float(item.get("time", 0.0)),
            x=float(item.get("x", 0.0)),
            speed_state=normalize_speed_state(item.get("speed_state", "normal")),
            object_id=item.get("object_id"),
            source=item.get("source", "fixture"),
        )
        for item in payload.get("speed_objects", [])
        if isinstance(item, dict)
    ]
    samples = [
        TimeXSample(
            time=_optional_float(item.get("time")),
            expected_x=_optional_float(item.get("expected_x")),
            x=_optional_float(item.get("x")),
            expected_time=_optional_float(item.get("expected_time")),
            tolerance=_optional_float(item.get("tolerance")),
            source=str(item.get("source", payload.get("source", "synthetic_approximate"))),
            notes=str(item.get("notes", "")),
        )
        for item in payload.get("samples", [])
        if isinstance(item, dict)
    ]
    return TimeXFixture(
        name=str(payload.get("name", fallback_name)),
        start_speed=normalize_speed_state(payload.get("start_speed", "normal")),
        song_offset=float(payload.get("song_offset", 0.0)),
        speed_objects=speed_objects,
        samples=samples,
        source=str(payload.get("source", "synthetic_approximate")),
        notes=str(payload.get("notes", "")),
    )


def _optional_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    try:
        if isinstance(value, (str, bytes, bytearray)):
            return float(value)
        if hasattr(value, "__float__"):
            return float(value) # type: ignore
        if hasattr(value, "__index__"):
            return float(value) # type: ignore
        return None
    except (TypeError, ValueError):
        return None


def _float_or_default(value: object, default: float) -> float:
    parsed = _optional_float(value)
    return default if parsed is None else parsed


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
