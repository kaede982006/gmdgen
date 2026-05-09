# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
from pathlib import Path

from gmdgen.validation.data_cleaning import inspect_dataset_gmd_files
from gmdgen.validation.pattern_schema import (
    validate_pattern_library,
    validate_pattern_object_payload,
    validate_patterns_index_payload,
)


def _pattern(pattern_id: str = "cube-easy-00") -> dict:
    return {
        "id": pattern_id,
        "mode": "cube",
        "difficulty": "easy",
        "length_beats": 8,
        "objects": [
            {"id": "1", "x_beat": 0.0, "y": 105, "role": "structural"},
            {"id": "8", "x_beat": 3.0, "y": 135, "role": "gameplay"},
        ],
        "entry": {"x_beat": 0.0, "y": 105, "speed": 1.0},
        "exit": {"x_beat": 8.0, "y": 105, "speed": 1.0},
        "tested": True,
        "source": "unit",
    }


def test_pattern_object_schema_accepts_valid_pattern() -> None:
    errors, warnings = validate_pattern_object_payload(_pattern())

    assert errors == []
    assert warnings == []


def test_patterns_index_schema_is_separate_from_object_pattern_schema() -> None:
    payload = {
        "version": 1,
        "cells": {"cube/easy": ["cube-easy-00"]},
        "patterns": {"cube-easy-00": _pattern()},
    }

    errors, warnings = validate_patterns_index_payload(payload, path="patterns_index.json")

    assert errors == []
    assert warnings == []


def test_invalid_pattern_does_not_poison_index_file_classification(tmp_path: Path) -> None:
    index = {
        "version": 1,
        "cells": {"cube/easy": ["cube-easy-00"]},
        "patterns": {"cube-easy-00": _pattern()},
    }
    (tmp_path / "patterns_index.json").write_text(json.dumps(index), encoding="utf-8")
    (tmp_path / "bad.json").write_text(json.dumps({**_pattern("bad"), "objects": []}), encoding="utf-8")

    report = validate_pattern_library(tmp_path).to_dict()

    assert report["checked_files"] == 2
    assert report["index_files"] == 1
    assert report["valid_patterns"] == 0
    assert len(report["invalid_patterns"]) == 1
    assert "bad.json" in report["invalid_patterns"][0]["path"]


def test_dataset_cleaning_is_report_first_and_non_destructive(tmp_path: Path) -> None:
    bad_file = tmp_path / "user_level.gmd"
    bad_file.write_text("not a valid gmd", encoding="utf-8")

    report = inspect_dataset_gmd_files(tmp_path)

    assert bad_file.exists()
    assert report.destructive_changes is False
    assert report.checked_files == 1
    assert report.invalid_files
