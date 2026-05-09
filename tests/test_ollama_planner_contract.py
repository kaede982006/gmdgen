# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from gmdgen.ai.ollama_provider import OllamaInvalidSchema, OllamaMissingRequiredField, OllamaProvider
from gmdgen.ai.planner import parse_ollama_section_plan, parse_or_fallback_planner_output
from gmdgen.ai.schemas import AILevelPlanResponse
from gmdgen.audio.analysis import AudioAnalysisResult
from gmdgen.generate.audio_conditioned import _audit_ollama_context_for_legacy_symbols, _maybe_apply_ai_provider


def _valid_payload() -> dict:
    return {
        "level_plan": {
            "level_name": "Unit Plan",
            "difficulty": "normal",
            "target_duration": 30.0,
            "object_budget": 500,
            "style": "modern_glow",
            "sync_intensity": "medium",
        },
        "sections": [
            {
                "section_id": "s001",
                "time_start": 0.0,
                "time_end": 8.0,
                "game_mode": "cube",
                "speed": "1x",
                "density": 0.35,
                "primary_pattern": "intro_platforming",
                "allowed_object_families": ["block", "spike", "orb", "pad"],
                "forbidden_features": ["unbounded_trigger_spam"],
                "trigger_budget": 3,
                "group_symbols": ["intro_blocks"],
                "design_notes": "short readable intro",
            }
        ],
    }


def test_ollama_planner_rejects_raw_gmd_output() -> None:
    result = parse_ollama_section_plan("1,100,2,30,3,90;1,101,2,60,3,120")

    assert result.plan is None
    assert "raw_gmd_output_rejected" in result.errors


def test_ollama_planner_rejects_concrete_group_and_color_ids() -> None:
    payload = _valid_payload()
    payload["sections"][0]["group_id"] = 17
    payload["sections"][0]["color_channel_id"] = 3

    result = parse_ollama_section_plan(payload)

    assert result.plan is None
    assert any("group_id:forbidden_planner_field" in error for error in result.errors)
    assert any("color_channel_id:forbidden_planner_field" in error for error in result.errors)


def test_schema_mismatch_uses_template_fallback_and_records_report_fields() -> None:
    result = parse_or_fallback_planner_output(
        {"sections": [{"section_id": "s001", "game_mode": "not-a-mode"}]},
        prompt="make a readable intro",
        fallback_level_name="fallback",
        object_budget=123,
    )

    assert result.plan is not None
    assert result.fallback_used is True
    assert result.plan.object_budget == 123
    fields = result.to_report_fields()
    assert fields["planner_fallback_used"] is True
    assert fields["planner_status"] == "fallback"


def test_valid_planner_output_uses_symbolic_references_only() -> None:
    result = parse_ollama_section_plan(_valid_payload())

    assert result.valid is True
    assert result.plan is not None
    section = result.plan.sections[0]
    assert section.group_symbols[0].name == "intro_blocks"
    assert not hasattr(section.group_symbols[0], "id")


def test_ollama_provider_rejects_legacy_object_plan_response() -> None:
    provider = OllamaProvider(
        model="unit-test",
        client=lambda _payload: {
            "response": json.dumps(
                {
                    "sections": [],
                    "object_plans": [{"object_id": 1, "x": 30, "y": 90, "role": "ai_structure"}],
                    "trigger_plans": [],
                }
            )
        },
        max_retries=0,
    )

    with pytest.raises(OllamaInvalidSchema):
        provider.generate_level_plan({"project_goal": "legacy output must be rejected"})


def test_ollama_provider_repairs_forbidden_fields_once() -> None:
    calls = {"count": 0}

    def client(_payload: dict) -> dict:
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "response": json.dumps(
                    {
                        "level_plan": {
                            "level_name": "Bad",
                            "difficulty": "normal",
                            "target_duration": 30.0,
                            "object_budget": 500,
                            "style": "modern",
                            "sync_intensity": "medium",
                        },
                        "sections": [
                            {
                                "section_id": "s001",
                                "time_start": 0.0,
                                "time_end": 8.0,
                                "game_mode": "cube",
                                "speed": "1x",
                                "density": 0.35,
                                "primary_pattern": "intro",
                                "object_plans": [],
                                "score": 1.0,
                            }
                        ],
                    }
                )
            }
        return {"response": json.dumps(_valid_payload())}

    provider = OllamaProvider(model="unit-test", client=client, max_retries=0)

    response = provider.generate_level_plan({"project_goal": "repair forbidden"})

    assert calls["count"] == 2
    assert response.metadata["planner_status"] == "success_repaired"
    report = response.metadata["planner_report"]
    assert report["forbidden_fields"] == ["object_plans", "score"]
    assert report["forbidden_field_paths"]


def test_ollama_provider_repairs_missing_sections_shape_once() -> None:
    calls = {"count": 0}

    repaired_payload = {
        "level_plan": {
            "level_name": "Stereo Madness 198",
            "difficulty": "easy",
            "target_duration": 198.0,
            "object_budget": 3000,
            "style": "classic_minimal",
            "sync_intensity": "medium",
        },
        "sections": [
            {
                "section_id": "s001",
                "time_start": 0.0,
                "time_end": 20.0,
                "game_mode": "cube",
                "speed": "1x",
                "density": 0.25,
                "primary_pattern": "simple_cube_intro",
                "allowed_object_families": ["block", "spike", "orb", "pad"],
                "forbidden_features": ["glow_spam"],
                "trigger_budget": 0,
                "group_symbols": [],
                "design_notes": "intro",
            }
        ],
    }

    def client(_payload: dict) -> dict:
        calls["count"] += 1
        if calls["count"] == 1:
            return {"response": json.dumps({"level_plan": {"sections": []}})}
        return {"response": json.dumps(repaired_payload)}

    provider = OllamaProvider(model="unit-test", client=client, max_retries=0)
    response = provider.generate_level_plan({"project_goal": "repair missing sections"})
    report = response.metadata["planner_report"]

    assert calls["count"] == 2
    assert response.metadata["planner_status"] == "success_repaired"
    assert response.metadata["planner_repair_attempted"] is True
    assert response.metadata["planner_repair_success"] is True
    assert report["repair_prompt_sent"] is True
    assert report["repair_success"] is True


def test_ollama_provider_schema_repair_failure_keeps_fallback_reason_clear() -> None:
    calls = {"count": 0}

    def client(_payload: dict) -> dict:
        calls["count"] += 1
        return {"response": json.dumps({"level_plan": {"sections": []}})}

    provider = OllamaProvider(model="unit-test", client=client, max_retries=0)
    with pytest.raises(OllamaMissingRequiredField):
        provider.generate_level_plan({"project_goal": "repair should fail"})

    assert calls["count"] == 2
    diag = provider.last_response_diagnostics
    assert diag["repair_prompt_sent"] is True
    assert diag["repair_success"] is False
    assert "$.sections" in diag["missing_required_fields"]


def test_ollama_provider_repairs_when_top_level_sections_empty() -> None:
    calls = {"count": 0}

    repaired_payload = {
        "level_plan": {
            "level_name": "Recovered",
            "difficulty": "normal",
            "target_duration": 30.0,
            "object_budget": 500,
            "style": "classic",
            "sync_intensity": "medium",
        },
        "sections": [
            {
                "section_id": "s001",
                "time_start": 0.0,
                "time_end": 10.0,
                "game_mode": "cube",
                "speed": "1x",
                "density": 0.3,
                "primary_pattern": "intro",
                "allowed_object_families": ["block", "spike", "orb", "pad"],
                "forbidden_features": [],
                "trigger_budget": 0,
                "group_symbols": [],
                "design_notes": "",
            }
        ],
    }

    def client(_payload: dict) -> dict:
        calls["count"] += 1
        if calls["count"] == 1:
            return {"response": json.dumps({"level_plan": repaired_payload["level_plan"], "sections": []})}
        return {"response": json.dumps(repaired_payload)}

    provider = OllamaProvider(model="unit-test", client=client, max_retries=0)
    response = provider.generate_level_plan({"project_goal": "repair empty sections"})

    assert calls["count"] == 2
    assert response.metadata["planner_status"] == "success_repaired"


def test_production_ollama_path_falls_back_on_legacy_object_plans() -> None:
    features = MagicMock(spec=AudioAnalysisResult)
    features.sections = []
    features.beat_times = []
    features.onset_times = []
    features.duration = 10.0
    features.bpm = 120.0
    features.confidence = 0.8
    features.confidence_report = None
    features.beat_features = []

    mock_provider = MagicMock()
    mock_provider.generate_level_plan.return_value = AILevelPlanResponse(
        object_plans=[{"object_id": 1, "x": 30, "y": 90, "role": "ai_structure"}],
        trigger_plans=[],
        provider="ollama",
        model="legacy-mock",
    )

    with patch("gmdgen.generate.audio_conditioned.create_ai_provider_from_config", return_value=mock_provider):
        conversion, metadata = _maybe_apply_ai_provider(
            config={"use_ai_planner": True, "require_ai_planning": False, "ai_candidate_count": 1},
            features=features,
            section_plans=[],
            time_x_report={},
            style_profile={},
            object_budget=100,
            max_group_id=999,
            safe_mode=True,
            start_speed="normal",
            song_offset=0.0,
        )

    assert conversion.response is not None
    assert conversion.response.provider == "local"
    assert conversion.response.fallback_used is True
    assert metadata["deterministic_fallback_used"] is True
    assert metadata["ai_planning_error"] == "ai_output_invalid"


def test_planner_alias_normalization() -> None:
    payload = {
        "level_plan": {
            "level_name": "Test Alias",
            "difficulty": "normal gameplay",
            "target_duration": 30.0,
            "object_budget": 500,
            "style": "modern_glow",
            "sync_intensity": "medium",
        },
        "sections": [
            {
                "section_id": "s001",
                "time_start": 0.0,
                "time_end": 8.0,
                "game_mode": "cube gameplay",
                "speed": "2.0",
                "target_density": 0.35,
                "primary_pattern": "intro_platforming",
                "allowed_objects": ["block", "spike"],
                "forbidden_features": ["glow_spam"],
                "trigger_budget": 3,
                "group_symbols": ["group_1"],
                "notes": "minimal decoration",
            }
        ],
    }
    
    result = parse_ollama_section_plan(payload)
    assert result.valid is True
    assert result.plan is not None
    assert result.plan.difficulty == "normal"
    section = result.plan.sections[0]
    assert section.game_mode == "cube"
    assert section.speed == "2x"
    assert section.density == 0.35
    assert section.allowed_object_families == ["block", "spike"]
    assert section.design_notes == "minimal decoration"

def test_planner_json_extraction_with_fence() -> None:
    from gmdgen.ai.ollama_provider import extract_json_object
    raw = """
Here is your plan:
```json
{
  "level_plan": {"level_name": "Test Extraction", "difficulty": "normal", "target_duration": 198.0, "object_budget": 3000, "style": "classic", "sync_intensity": "low"},
  "sections": [
    {
      "section_id": "s001", "time_start": 0.0, "time_end": 20.0, "game_mode": "cube", "speed": "1x", "density": 0.25,
      "primary_pattern": "simple_cube", "allowed_object_families": ["block"], "forbidden_features": [],
      "trigger_budget": 0, "group_symbols": [], "design_notes": "minimal"
    }
  ]
}
```
Good luck!
"""
    extracted = extract_json_object(raw)
    assert "level_plan" in extracted
    assert extracted["level_plan"]["level_name"] == "Test Extraction"


def test_ollama_context_audit_detects_legacy_symbols() -> None:
    audit = _audit_ollama_context_for_legacy_symbols(
        [
            {"path": "docs/old.md", "text": "Legacy ObjectPlan and TriggerPlan example"},
            {"path": "dataset/raw.txt", "text": "raw_gmd save_string example"},
        ]
    )

    assert audit["symbols"] == ["ObjectPlan", "TriggerPlan", "raw_gmd", "save_string"]
    assert "docs/old.md" in audit["paths"]
    assert "dataset/raw.txt" in audit["paths"]
