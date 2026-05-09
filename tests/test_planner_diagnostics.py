# SPDX-License-Identifier: GPL-3.0-or-later
import json
import pytest
from gmdgen.ai.planner import parse_ollama_section_plan, PlannerParseResult

def test_planner_missing_fields_defaults():
    # Test that missing optional fields are now handled with defaults
    payload = {
        "level_plan": {
            "level_name": "Test",
            "difficulty": "normal",
            "target_duration": 30.0,
            "object_budget": 1000,
            "style": "modern",
            "sync_intensity": "medium"
        },
        "sections": [
            {
                "section_id": "s001",
                "time_start": 0.0,
                "time_end": 10.0,
                "game_mode": "cube",
                "speed": "1x",
                "density": 0.5,
                "primary_pattern": "test_pattern",
            }
        ]
    }
    result = parse_ollama_section_plan(payload)
    assert result.valid
    assert result.plan.sections[0].trigger_budget == 0
    assert result.plan.sections[0].design_notes == ""
    assert result.plan.sections[0].allowed_object_families == ["block", "spike", "orb", "pad"]

def test_planner_forbidden_fields():
    payload = {
        "level_plan": {"level_name": "Test", "difficulty": "normal", "target_duration": 30.0, "object_budget": 1000, "style": "modern", "sync_intensity": "medium"},
        "sections": [
            {
                "section_id": "s001", "time_start": 0.0, "time_end": 10.0, "game_mode": "cube", "speed": "1x", "density": 0.5, "primary_pattern": "test",
                "object_plans": [] # FORBIDDEN
            }
        ]
    }
    result = parse_ollama_section_plan(payload)
    assert not result.valid
    assert any("forbidden_planner_field" in e for e in result.errors)

def test_planner_alias_normalization():
    payload = {
        "plan": { # top-level alias
            "level_name": "Test",
            "difficulty": "Normal Difficulty", # value alias
            "target_duration": 30.0,
            "object_budget": 1000,
            "style": "modern",
            "sync_intensity": "Moderate" # value alias
        },
        "sections": [
            {
                "section_id": "s001",
                "time_start": 0.0,
                "time_end": 10.0,
                "game_mode": "Cube Gameplay", # value alias
                "speed": "Normal Speed", # value alias
                "target_density": 0.5, # key alias
                "primary_pattern": "test",
                "allowed_objects": ["block"], # key alias
                "forbidden": ["nothing"], # key alias
                "notes": "test notes" # key alias
            }
        ]
    }
    result = parse_ollama_section_plan(payload)
    assert result.valid
    assert result.plan.difficulty == "normal"
    assert result.plan.sync_intensity == "medium"
    assert result.plan.sections[0].game_mode == "cube"
    assert result.plan.sections[0].speed == "1x"
    assert result.plan.sections[0].density == 0.5
    assert result.plan.sections[0].allowed_object_families == ["block"]
    assert result.plan.sections[0].forbidden_features == ["nothing"]
    assert result.plan.sections[0].design_notes == "test notes"

def test_planner_diagnostics_in_report():
    payload = {"invalid": "json"}
    result = parse_ollama_section_plan(payload)
    report = result.to_report_fields()
    assert report["planner_status"] == "invalid"
    assert "planner_raw_payload_preview" in report
    assert report["planner_raw_payload_preview"] is not None


def test_planner_report_records_forbidden_field_diagnostics():
    payload = {
        "level_plan": {
            "level_name": "Test",
            "difficulty": "normal",
            "target_duration": 30.0,
            "object_budget": 1000,
            "style": "modern",
            "sync_intensity": "medium",
        },
        "sections": [
            {
                "section_id": "s001",
                "time_start": 0.0,
                "time_end": 10.0,
                "game_mode": "cube",
                "speed": "1x",
                "density": 0.5,
                "primary_pattern": "test",
                "object_plans": [],
                "score": 1.0,
            }
        ],
    }

    report = parse_ollama_section_plan(payload).to_report_fields()

    assert report["forbidden_fields"] == ["object_plans", "score"]
    assert "$.sections[0].object_plans" in report["forbidden_field_paths"]
    assert report["schema_error_path"] == "$.sections[0].object_plans"
    assert report["raw_ollama_response_preview"]
    assert report["extracted_json_preview"]


def _valid_section(section_id: str, start: float, end: float, mode: str = "cube") -> dict:
    return {
        "section_id": section_id,
        "time_start": start,
        "time_end": end,
        "game_mode": mode,
        "speed": "1x",
        "density": 0.3,
        "primary_pattern": f"pattern_{section_id}",
        "allowed_object_families": ["block", "spike", "orb", "pad"],
        "forbidden_features": ["glow_spam"],
        "trigger_budget": 0,
        "group_symbols": [],
        "design_notes": "test",
    }


def test_nested_sections_shape_is_normalized_to_top_level() -> None:
    payload = {
        "level_plan": {
            "level_name": "x",
            "difficulty": "easy",
            "target_duration": 198.0,
            "object_budget": 3000,
            "style": "classic_minimal",
            "sync_intensity": "medium",
            "sections": [_valid_section("s001", 0.0, 20.0)],
        }
    }
    result = parse_ollama_section_plan(payload)
    report = result.to_report_fields()
    assert result.valid
    assert report["planner_status"] == "success_normalized"
    assert "$.level_plan.sections" in report["wrong_location_fields"]
    assert any("moved_$.level_plan.sections_to_$.sections" in item for item in report["normalized_shape_repairs"])


def test_empty_nested_sections_reports_missing_and_empty() -> None:
    payload = {"level_plan": {"sections": []}}
    report = parse_ollama_section_plan(payload).to_report_fields()
    assert report["planner_status"] == "invalid"
    assert "$.sections" in report["missing_required_fields"]
    assert "$.level_plan.sections" in report["empty_required_fields"]


def test_missing_top_level_sections_reports_missing_required() -> None:
    payload = {"level_plan": {"level_name": "x"}}
    report = parse_ollama_section_plan(payload).to_report_fields()
    assert report["planner_status"] == "invalid"
    assert "$.sections" in report["missing_required_fields"]


def test_top_level_sections_empty_reports_empty_required() -> None:
    payload = {
        "level_plan": {
            "level_name": "x",
            "difficulty": "easy",
            "target_duration": 198.0,
            "object_budget": 3000,
            "style": "classic_minimal",
            "sync_intensity": "medium",
        },
        "sections": [],
    }
    report = parse_ollama_section_plan(payload).to_report_fields()
    assert report["planner_status"] == "invalid"
    assert "$.sections" in report["empty_required_fields"]


def test_valid_198s_planner_json_covers_full_duration() -> None:
    payload = {
        "level_plan": {
            "level_name": "Stereo Madness 198",
            "difficulty": "easy",
            "target_duration": 198.0,
            "object_budget": 3000,
            "style": "classic_minimal",
            "sync_intensity": "medium",
        },
        "sections": [
            _valid_section("s001", 0.0, 20.0, "cube"),
            _valid_section("s002", 20.0, 45.0, "cube"),
            _valid_section("s003", 45.0, 70.0, "ship"),
            _valid_section("s004", 70.0, 95.0, "cube"),
            _valid_section("s005", 95.0, 120.0, "ball"),
            _valid_section("s006", 120.0, 145.0, "cube"),
            _valid_section("s007", 145.0, 170.0, "cube"),
            _valid_section("s008", 170.0, 198.0, "cube"),
        ],
    }
    result = parse_ollama_section_plan(payload)
    report = result.to_report_fields()
    assert result.valid
    assert report["planner_status"] in {"success", "success_normalized"}
    assert result.plan.sections[0].time_start == 0.0
    assert result.plan.sections[-1].time_end == 198.0
