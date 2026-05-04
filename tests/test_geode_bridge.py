# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from gmdgen.gd.geode_bridge import (
    NullGeodeBridge,
    OptionalGeodeFixtureBridge,
    compare_time_mapping_with_geode,
)
from gmdgen.gd.time_mapping import pos_for_time_like_gd


def test_null_geode_bridge_unavailable() -> None:
    bridge = NullGeodeBridge()

    assert bridge.is_available() is False
    report = bridge.validate_level_string("1,1,2,0,3,0;")
    assert report.available is False
    assert "geode_unavailable" in report.warnings


def test_geode_fixture_bridge_time_x_parity() -> None:
    fixture = {
        "geode_version": "fixture",
        "samples": [
            {"time": 0.0, "x": pos_for_time_like_gd(0.0), "expected_x": pos_for_time_like_gd(0.0), "expected_time": 0.0},
            {"time": 1.0, "x": pos_for_time_like_gd(1.0), "expected_x": pos_for_time_like_gd(1.0), "expected_time": 1.0},
        ],
    }
    bridge = OptionalGeodeFixtureBridge(fixture)

    report = compare_time_mapping_with_geode(bridge, [0.0, 1.0], [], "normal", 0.0)

    assert report.available is True
    assert report.checked is True
    assert report.passed is True
    assert report.sample_count == 2


def test_geode_unavailable_does_not_crash_generation() -> None:
    report = compare_time_mapping_with_geode(NullGeodeBridge(), [0.0, 1.0], [], "normal", 0.0)

    assert report.available is False
    assert report.checked is False
    assert "geode_unavailable_using_python_approximation" in report.warnings


def test_geode_parity_mismatch_records_warning() -> None:
    fixture = {
        "samples": [
            {"time": 1.0, "x": pos_for_time_like_gd(1.0), "expected_x": pos_for_time_like_gd(1.0) + 10.0, "expected_time": 1.2}
        ]
    }
    bridge = OptionalGeodeFixtureBridge(fixture)

    report = compare_time_mapping_with_geode(bridge, [1.0], [], "normal", 0.0, tolerance=0.01)

    assert report.passed is False
    assert "geode_time_x_parity_mismatch" in report.warnings
