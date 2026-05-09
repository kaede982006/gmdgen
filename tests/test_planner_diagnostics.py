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
