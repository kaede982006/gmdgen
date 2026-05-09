# SPDX-License-Identifier: GPL-3.0-or-later
import json
import pytest
from gmdgen.ai.planner import parse_ollama_section_plan

def test_stereo_madness_198s_plan():
    # Mocking a realistic response for a 198s Stereo Madness style level
    payload = {
        "level_plan": {
            "level_name": "Stereo Madness Tribute",
            "difficulty": "Easy",
            "target_duration": 198.0,
            "object_budget": 5000,
            "style": "classic",
            "sync_intensity": "Low"
        },
        "sections": [
            {
                "section_id": "intro",
                "time_start": 0.0,
                "time_end": 20.0,
                "game_mode": "Cube",
                "speed": "1x",
                "density": 0.2,
                "primary_pattern": "basic_jumping",
                "allowed_object_families": ["block", "spike"],
                "design_notes": "Very simple intro"
            },
            {
                "section_id": "ship_part",
                "time_start": 20.0,
                "time_end": 40.0,
                "game_mode": "Ship",
                "speed": "1x",
                "density": 0.3,
                "primary_pattern": "straight_fly",
                "allowed_object_families": ["block", "spike"],
                "design_notes": "Simple ship fly"
            },
            {
                "section_id": "final",
                "time_start": 40.0,
                "time_end": 198.0,
                "game_mode": "Cube",
                "speed": "1x",
                "density": 0.4,
                "primary_pattern": "ending_sequence",
                "allowed_object_families": ["block", "spike", "orb"],
                "design_notes": "Ending"
            }
        ]
    }
    
    result = parse_ollama_section_plan(payload)
    assert result.valid
    assert result.plan.level_name == "Stereo Madness Tribute"
    assert result.plan.target_duration == 198.0
    assert len(result.plan.sections) == 3
    assert result.plan.sections[2].time_end == 198.0
    # Check defaults
    assert result.plan.sections[0].forbidden_features == []
    assert result.plan.sections[0].trigger_budget == 0
