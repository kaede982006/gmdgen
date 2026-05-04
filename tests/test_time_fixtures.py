# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import csv
import json
from pathlib import Path

from gmdgen.gd.time_fixtures import (
    compare_with_time_x_fixtures,
    load_time_x_fixture,
    load_time_x_fixtures,
    summarize_time_x_fixture_errors,
)


FIXTURE_PATH = Path("tests/fixtures/time_mapping/example_time_x_fixture.json")


def test_load_time_x_fixture_json() -> None:
    fixture = load_time_x_fixture(FIXTURE_PATH)
    assert fixture.name == "synthetic_normal_speed_fixture"
    assert fixture.source == "synthetic_approximate"
    assert fixture.samples


def test_compare_with_time_x_fixtures_synthetic() -> None:
    fixture = load_time_x_fixture(FIXTURE_PATH)
    results = compare_with_time_x_fixtures([fixture])
    assert results[0].passed is True
    assert results[0].sample_count == 3


def test_time_x_fixture_report_contains_error_stats() -> None:
    fixture = load_time_x_fixture(FIXTURE_PATH)
    summary = summarize_time_x_fixture_errors(compare_with_time_x_fixtures([fixture]))
    assert summary["fixture_count"] == 1
    assert "average_abs_x_error" in summary
    assert "max_abs_time_error" in summary


def test_time_x_fixture_missing_optional_fields_safe(tmp_path: Path) -> None:
    path = tmp_path / "minimal.json"
    path.write_text(
        json.dumps({"name": "minimal", "samples": [{"time": 0.0, "expected_x": 0.0}]}),
        encoding="utf-8",
    )
    fixture = load_time_x_fixture(path)
    result = compare_with_time_x_fixtures([fixture])[0]
    assert result.passed is True


def test_time_x_fixture_future_geode_format_supported(tmp_path: Path) -> None:
    path = tmp_path / "geode.json"
    path.write_text(
        json.dumps(
            {
                "time_x_fixture": {
                    "name": "future_geode_export",
                    "source": "geode_leveltools",
                    "start_speed": "normal",
                    "samples": [{"time": 1.0, "expected_x": 311.58, "tolerance": 0.001}],
                }
            }
        ),
        encoding="utf-8",
    )
    fixture = load_time_x_fixture(path)
    result = compare_with_time_x_fixtures([fixture])[0]
    assert fixture.source == "geode_leveltools"
    assert result.passed is True


def test_load_time_x_fixture_csv(tmp_path: Path) -> None:
    path = tmp_path / "fixture.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fixture_name",
                "start_speed",
                "song_offset",
                "time",
                "expected_x",
                "tolerance",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "fixture_name": "csv_fixture",
                "start_speed": "normal",
                "song_offset": "0",
                "time": "1",
                "expected_x": "311.58",
                "tolerance": "0.001",
                "source": "synthetic_approximate",
            }
        )
    fixtures = load_time_x_fixtures(tmp_path)
    assert len(fixtures) == 1
    assert compare_with_time_x_fixtures(fixtures)[0].passed is True
