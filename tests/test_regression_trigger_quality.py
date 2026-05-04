from __future__ import annotations

from gmdgen.ai.schemas import convert_ai_response_to_plans, parse_ai_level_plan_response
from gmdgen.gd.plans import SectionPlan
from gmdgen.gd.time_mapping import SpeedState
from gmdgen.generate.quality_gate import QualityGateThresholds, evaluate_quality_gate


def _sections() -> list[SectionPlan]:
    return [
        SectionPlan(0, 2, 0, 480, "intro", "cube", SpeedState.NORMAL, 0.25, 0.2, 0.1, 0.2),
        SectionPlan(2, 4, 480, 960, "drop", "cube", SpeedState.NORMAL, 0.9, 0.8, 0.8, 0.7),
    ]


def test_regression_trigger_pruning_missing_group_playability() -> None:
    response = parse_ai_level_plan_response(
        {
            "sections": [],
            "object_plans": [
                {"object_id": 500, "x": 520, "y": 240, "role": "visual_accent_target"},
                {"object_id": 36, "x": 560, "y": 180, "role": "orb"},
            ],
            "trigger_plans": [
                {
                    "trigger_type": "pulse",
                    "object_id": "1006",
                    "x": 520,
                    "y": 300,
                    "target_group": None,
                    "duration": None,
                    "properties": {
                        "move_x": 24,
                        "move_y": 12,
                        "opacity": 0.5,
                        "color_channel": 2,
                        "trigger_kind": "pulse",
                        "purpose": "drop_accent",
                        "target_role": "decoration_group",
                        "intensity": 0.9,
                        "duration_hint": 0.16,
                        "section_id": 1,
                    },
                },
                {
                    "trigger_type": "move",
                    "object_id": "901",
                    "x": 560,
                    "y": 300,
                    "target_group": None,
                    "duration": 0.2,
                    "easing": "ease-in-out",
                    "properties": {
                        "color_channel": 3,
                        "opacity": 0.8,
                        "move_x": 12,
                        "purpose": "beat_accent",
                        "target_role": "gameplay_group",
                        "section_id": 1,
                    },
                },
            ],
        }
    )

    converted = convert_ai_response_to_plans(
        response,
        object_budget=20,
        max_group_id=100,
        safe_mode=True,
        section_plans=_sections(),
    )

    assert converted.valid
    assert converted.normalization_report.pruned_trigger_property_count == 0
    assert converted.normalization_report.ignored_irrelevant_trigger_property_count >= 3
    assert converted.normalization_report.auto_assigned_target_group_count == 2
    assert all(trigger.target_group is not None for trigger in converted.trigger_plans)


def test_regression_quality_gate_failure_actionable() -> None:
    result = evaluate_quality_gate(
        {
            "score": 0.2,
            "final_object_count": 4,
            "removed_object_ratio": 0.1,
            "removed_trigger_ratio": 0.7,
            "drop_impact_score": 0.1,
            "metrics": {"density_alignment_score": 0.1, "unresolved_missing_target_group_count": 2},
            "score_breakdown": {"editor_validity": 0.9, "playability_safety": 0.3},
            "pruned_trigger_property_count": 12,
            "repair_quality_report": {"removed_due_to_missing_target_group": 2},
            "playability_breakdown": {"spacing_score": 0.2, "input_density_score": 0.3},
            "density_target_by_section": {"1": 0.9},
            "actual_density_by_section": {"1": 0.1},
        },
        QualityGateThresholds(min_object_count=10),
    )

    assert result.passed is False
    assert result.can_retry is True
    assert result.can_regenerate_weak_sections is True
    assert result.trigger_pruning_count == 12
    assert result.missing_target_group_count == 2
    assert result.retry_prompt_summary
    assert any("trigger intents" in action for action in result.recommended_actions)
