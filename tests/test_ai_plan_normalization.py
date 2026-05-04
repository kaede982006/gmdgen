from __future__ import annotations

from gmdgen.ai.normalization import normalize_easing, normalize_object_role
from gmdgen.ai.schemas import parse_ai_level_plan_response, convert_ai_response_to_plans
from gmdgen.gd.plans import SectionPlan
from gmdgen.gd.time_mapping import SpeedState


def _section() -> SectionPlan:
    return SectionPlan(
        start_time=0.0,
        end_time=4.0,
        start_x=0.0,
        end_x=900.0,
        section_type="normal",
        gameplay_mode="cube",
        speed_state=SpeedState.NORMAL,
        density_target=0.5,
        decoration_intensity=0.5,
        trigger_intensity=0.5,
        difficulty_target=0.5,
    )


def test_ai_output_with_orb_and_jump_pad_is_normalized() -> None:
    assert normalize_object_role("orb") == ("beat_orb", True)
    assert normalize_object_role("jump_pad") == ("beat_pad", True)

    response = parse_ai_level_plan_response(
        {
            "sections": [],
            "object_plans": [
                {"object_id": 36, "x": 100, "y": 120, "role": "orb"},
                {"object_id": 35, "x": 180, "y": 90, "role": "jump_pad"},
            ],
            "trigger_plans": [],
        }
    )

    converted = convert_ai_response_to_plans(
        response,
        object_budget=10,
        max_group_id=100,
        safe_mode=True,
        section_plans=[_section()],
    )

    assert converted.valid
    assert [plan.role for plan in converted.object_plans] == ["beat_orb", "beat_pad"]
    assert converted.normalization_report.normalized_object_role_count == 2


def test_ai_trigger_irrelevant_properties_are_pruned() -> None:
    response = parse_ai_level_plan_response(
        {
            "sections": [],
            "object_plans": [{"object_id": 500, "x": 100, "y": 240, "role": "visual_accent_target", "group_ids": [1]}],
            "trigger_plans": [
                {
                    "trigger_type": "pulse",
                    "object_id": "1006",
                    "x": 100,
                    "y": 300,
                    "target_group": 1,
                    "duration": 0.1,
                    "properties": {"move_x": 12, "opacity": 0.4, "color_channel": 2},
                },
                {
                    "trigger_type": "color",
                    "object_id": "29",
                    "x": 120,
                    "y": 300,
                    "target_group": 1,
                    "duration": 0.1,
                    "properties": {"move_x": 9, "move_y": 4, "color_channel": 3},
                },
                {
                    "trigger_type": "move",
                    "object_id": "901",
                    "x": 140,
                    "y": 300,
                    "target_group": 1,
                    "duration": 0.1,
                    "properties": {"color_channel": 5, "move_x": 8},
                },
            ],
        }
    )

    converted = convert_ai_response_to_plans(
        response,
        object_budget=20,
        max_group_id=100,
        safe_mode=True,
        section_plans=[_section()],
    )

    assert converted.valid
    assert converted.normalization_report.pruned_trigger_property_count == 0
    assert converted.normalization_report.ignored_irrelevant_trigger_property_count >= 4
    assert all("unknown_trigger_property" not in warning for warning in converted.warnings)
    assert converted.trigger_plans[0].properties == {"color_channel": 2}
    assert converted.trigger_plans[1].properties == {"color_channel": 3}
    assert converted.trigger_plans[2].properties["move_x"] == 8
    assert "color_channel" not in converted.trigger_plans[2].properties


def test_ai_move_easing_is_normalized_or_rejected_cleanly() -> None:
    assert normalize_easing("easeInOut") == ("ease_in_out", True)
    assert normalize_easing("ease-in") == ("ease_in", True)
    assert normalize_easing("none") == ("linear", True)

    response = parse_ai_level_plan_response(
        {
            "sections": [],
            "object_plans": [{"object_id": 1, "x": 100, "y": 90, "role": "ai_structure", "group_ids": [1]}],
            "trigger_plans": [
                {
                    "trigger_type": "move",
                    "object_id": "901",
                    "x": 100,
                    "y": 300,
                    "target_group": 1,
                    "duration": 0.2,
                    "easing": "easeInOut",
                    "properties": {"move_x": 8},
                }
            ],
        }
    )

    converted = convert_ai_response_to_plans(
        response,
        object_budget=10,
        max_group_id=100,
        safe_mode=True,
        section_plans=[_section()],
    )

    assert converted.valid
    assert converted.trigger_plans[0].easing == "ease_in_out"
    assert converted.normalization_report.normalized_easing_count == 1


def test_ai_output_normalization_runs_before_validator() -> None:
    response = parse_ai_level_plan_response(
        {
            "sections": [],
            "object_plans": [{"object_id": 36, "x": 100, "y": 120, "role": "orb"}],
            "trigger_plans": [
                {
                    "trigger_type": "move",
                    "object_id": "901",
                    "x": 120,
                    "y": 300,
                    "target_group": 1,
                    "duration": 0.1,
                    "properties": {"easing": "ease-out", "color_channel": 9, "move_y": 4},
                }
            ],
        }
    )

    converted = convert_ai_response_to_plans(
        response,
        object_budget=10,
        max_group_id=100,
        safe_mode=True,
        section_plans=[_section()],
    )

    assert converted.valid
    assert converted.object_plans[0].role == "beat_orb"
    assert converted.trigger_plans[0].easing == "ease_out"
    assert converted.trigger_plans[0].properties["move_y"] == 4
    assert "color_channel" not in converted.trigger_plans[0].properties
