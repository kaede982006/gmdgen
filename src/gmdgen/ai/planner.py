# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from gmdgen.generate.ir import ColorSymbol, GroupSymbol, LevelPlan, SectionPlan


DIFFICULTIES = {"easy", "normal", "hard", "insane", "demon"}
GAME_MODES = {"cube", "ship", "ball", "ufo", "wave", "robot", "spider"}
SPEEDS = {"0.5x", "1x", "2x", "3x", "4x"}
SYNC_INTENSITIES = {"low", "medium", "high"}
FORBIDDEN_KEYS = {
    "raw_gmd",
    "gmd",
    "gmd_string",
    "save_string",
    "object_plans",
    "trigger_plans",
    "objects",
    "triggers",
    "group_id",
    "group_ids",
    "target_group",
    "secondary_group",
    "color_channel",
    "color_channel_id",
    "final_score",
    "score",
    "validation_passed",
    "playability_passed",
}
RAW_GMD_PATTERN = re.compile(r"(?:^|[;\s])1,\d+,2,-?\d+(?:\.\d+)?,3,-?\d+(?:\.\d+)?(?:[;,]|$)")


@dataclass(slots=True)
class PlannerParseResult:
    plan: LevelPlan | None = None
    errors: list[str] = field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: str = ""
    raw_payload: str | None = None
    json_payload: dict[str, Any] | None = None
    forbidden_fields: list[str] = field(default_factory=list)
    forbidden_field_paths: list[str] = field(default_factory=list)
    schema_error_path: str | None = None
    missing_required_fields: list[str] = field(default_factory=list)
    empty_required_fields: list[str] = field(default_factory=list)
    wrong_location_fields: list[str] = field(default_factory=list)
    schema_error_message: str | None = None
    planner_failure_stage: str | None = None
    planner_failure_reason_detail: str | None = None
    normalized_shape_repairs: list[str] = field(default_factory=list)
    repair_prompt_sent: bool = False
    repair_response_preview: str | None = None
    repair_success: bool | None = None
    planner_repair_skipped_reason: str = ""

    @property
    def valid(self) -> bool:
        return self.plan is not None and not self.errors and not self.fallback_used

    def to_report_fields(self) -> dict[str, Any]:
        extracted_preview = None
        if self.json_payload is not None:
            extracted_preview = json.dumps(self.json_payload, ensure_ascii=False, sort_keys=True)[:1000]
        planner_status = "invalid"
        if self.fallback_used:
            planner_status = "fallback"
        elif self.plan is not None:
            planner_status = "success_normalized" if self.normalized_shape_repairs else "success"
        return {
            "planner_status": planner_status,
            "planner_fallback_used": self.fallback_used,
            "planner_fallback_reason": self.fallback_reason,
            "planner_errors": list(self.errors),
            "planner_raw_payload_preview": str(self.raw_payload)[:1000] if self.raw_payload else None,
            "planner_json_payload": self.json_payload if self.plan else None,
            "raw_ollama_response_preview": str(self.raw_payload)[:1000] if self.raw_payload else None,
            "extracted_json_preview": extracted_preview,
            "forbidden_fields": list(self.forbidden_fields),
            "forbidden_field_paths": list(self.forbidden_field_paths),
            "schema_error_path": self.schema_error_path,
            "missing_required_fields": list(self.missing_required_fields),
            "empty_required_fields": list(self.empty_required_fields),
            "wrong_location_fields": list(self.wrong_location_fields),
            "schema_error_message": self.schema_error_message,
            "planner_failure_stage": self.planner_failure_stage,
            "planner_failure_reason_detail": self.planner_failure_reason_detail,
            "normalized_shape_repairs": list(self.normalized_shape_repairs),
            "repair_prompt_sent": self.repair_prompt_sent,
            "repair_response_preview": self.repair_response_preview,
            "repair_success": self.repair_success,
            "planner_repair_skipped_reason": self.planner_repair_skipped_reason,
        }


def parse_ollama_section_plan(payload: str | dict[str, Any]) -> PlannerParseResult:
    """Parse the strict Ollama planner JSON contract.

    Ollama is only allowed to describe a level and sections with symbolic
    references. Concrete Geometry Dash object strings, group ids, color channel
    ids, scores, and validation verdicts are rejected here.
    """
    raw_payload = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    normalized_shape_repairs: list[str] = []
    missing_required_fields: list[str] = []
    empty_required_fields: list[str] = []
    wrong_location_fields: list[str] = []

    if isinstance(payload, str):
        if _looks_like_raw_gmd(payload):
            return _result_with_diagnostics(
                errors=["raw_gmd_output_rejected"],
                raw_payload=raw_payload,
                json_payload=None,
                planner_failure_stage="shape_guard",
                planner_failure_reason_detail="raw .gmd style output is forbidden for Ollama planner JSON",
            )
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            return _result_with_diagnostics(
                errors=[f"json_parse_failed:{exc.msg}"],
                raw_payload=raw_payload,
                json_payload=None,
                planner_failure_stage="json_parse",
                planner_failure_reason_detail=f"json parse failed: {exc.msg}",
            )
    elif isinstance(payload, dict):
        data = payload
    else:
        return _result_with_diagnostics(
            errors=["planner_output_must_be_json_object"],
            raw_payload=raw_payload,
            json_payload=None,
            planner_failure_stage="json_parse",
            planner_failure_reason_detail="planner output is not a JSON object",
        )

    errors = _reject_forbidden_shape(data)
    if errors:
        return _result_with_diagnostics(
            errors=errors,
            raw_payload=raw_payload,
            json_payload=data if isinstance(data, dict) else None,
            planner_failure_stage="forbidden_field_check",
            planner_failure_reason_detail="forbidden planner keys were found in payload",
        )

    if not isinstance(data, dict):
        return _result_with_diagnostics(
            errors=["planner_output_must_be_json_object"],
            raw_payload=raw_payload,
            json_payload=None,
            planner_failure_stage="schema_validation",
            planner_failure_reason_detail="top-level JSON must be an object",
        )

    # Handle top-level aliases first so normalization can find them
    if "level_plan" not in data:
        for alias in ["plan", "level", "level_data", "levelPlan"]:
            if alias in data and isinstance(data[alias], dict):
                data["level_plan"] = data.pop(alias)
                normalized_shape_repairs.append(f"top_level_alias:{alias}->level_plan")
                break

    if "sections" not in data:
        for alias in ["section_list", "section_plans", "sections_plan"]:
            if alias in data and isinstance(data[alias], list):
                data["sections"] = data.pop(alias)
                normalized_shape_repairs.append(f"top_level_alias:{alias}->sections")
                break

    level_payload = data.get("level_plan")
    if isinstance(level_payload, dict):
        for nested_alias in ["sections", "section_list", "section_plans"]:
            if nested_alias in level_payload:
                wrong_location_fields.append(f"$.level_plan.{nested_alias}")
                nested_sections = level_payload.get(nested_alias)
                if "sections" not in data and isinstance(nested_sections, list):
                    if nested_sections:
                        data["sections"] = nested_sections
                        level_payload.pop(nested_alias, None)
                        normalized_shape_repairs.append(f"moved_$.level_plan.{nested_alias}_to_$.sections")
                    else:
                        empty_required_fields.append(f"$.level_plan.{nested_alias}")
                elif isinstance(nested_sections, list):
                    level_payload.pop(nested_alias, None)
                    normalized_shape_repairs.append(f"removed_$.level_plan.{nested_alias}_duplicate")

    _normalize_dict_aliases(data)

    allowed_top = {"level_plan", "sections"}
    unknown_top = set(data) - allowed_top
    if unknown_top:
        # We allow some unknown top keys but log them if they are not common noise
        noise = {"reasoning", "thoughts", "explanation", "plan", "level"}
        real_unknown = unknown_top - noise
        if real_unknown:
            errors.append(f"unknown_top_level_keys:{','.join(sorted(real_unknown))}")

    level_payload = data.get("level_plan")
    sections_payload = data.get("sections")
    if not isinstance(level_payload, dict):
        errors.append("level_plan_required")
        _append_unique(missing_required_fields, "$.level_plan")
    if not isinstance(sections_payload, list):
        errors.append("sections_required")
        _append_unique(missing_required_fields, "$.sections")
    elif len(sections_payload) == 0:
        errors.append("sections_empty")
        _append_unique(empty_required_fields, "$.sections")

    if not isinstance(level_payload, dict) or not isinstance(sections_payload, list):
        return _result_with_diagnostics(
            errors=errors,
            raw_payload=raw_payload,
            json_payload=data,
            missing_required_fields=missing_required_fields,
            empty_required_fields=empty_required_fields,
            wrong_location_fields=wrong_location_fields,
            normalized_shape_repairs=normalized_shape_repairs,
            planner_failure_stage="schema_validation",
            planner_failure_reason_detail=_planner_failure_reason_detail(
                missing_required_fields=missing_required_fields,
                empty_required_fields=empty_required_fields,
                wrong_location_fields=wrong_location_fields,
            ),
        )

    # Now level_payload is dict and sections_payload is list
    level_errors = _validate_level_plan_payload(level_payload)
    _collect_level_plan_missing_paths(level_errors, missing_required_fields)
    section_errors: list[str] = []
    sections: list[SectionPlan] = []
    for index, raw_section in enumerate(sections_payload):
        if not isinstance(raw_section, dict):
            section_errors.append(f"sections[{index}]:must_be_object")
            continue
        parsed, parsed_errors = _parse_section_payload(raw_section, index)
        section_errors.extend(parsed_errors)
        if parsed is not None:
            sections.append(parsed)

    errors.extend(level_errors)
    errors.extend(section_errors)
    if not sections:
        errors.append("sections_empty")
        _append_unique(empty_required_fields, "$.sections")

    if errors:
        _collect_section_missing_paths(section_errors, missing_required_fields)
        return _result_with_diagnostics(
            errors=errors,
            raw_payload=raw_payload,
            json_payload=data,
            missing_required_fields=missing_required_fields,
            empty_required_fields=empty_required_fields,
            wrong_location_fields=wrong_location_fields,
            normalized_shape_repairs=normalized_shape_repairs,
            planner_failure_stage="schema_validation",
            planner_failure_reason_detail=_planner_failure_reason_detail(
                missing_required_fields=missing_required_fields,
                empty_required_fields=empty_required_fields,
                wrong_location_fields=wrong_location_fields,
            ),
        )

    return PlannerParseResult(
        plan=LevelPlan(
            level_name=str(level_payload.get("level_name", "unnamed")),
            difficulty=str(level_payload.get("difficulty", "normal")).lower(),
            target_duration=float(level_payload.get("target_duration", 30.0)),
            object_budget=int(level_payload.get("object_budget", 1000)),
            style=str(level_payload.get("style", "modern")),
            sync_intensity=str(level_payload.get("sync_intensity", "medium")).lower(),
            sections=sections,
        ),
        raw_payload=raw_payload,
        json_payload=data,
        schema_error_path=None,
        normalized_shape_repairs=normalized_shape_repairs,
        missing_required_fields=missing_required_fields,
        empty_required_fields=empty_required_fields,
        wrong_location_fields=wrong_location_fields,
        planner_failure_stage=None,
        planner_failure_reason_detail="",
    )


def parse_or_fallback_planner_output(
    payload: str | dict[str, Any],
    *,
    prompt: str = "",
    fallback_level_name: str = "fallback_plan",
    object_budget: int = 500,
) -> PlannerParseResult:
    parsed = parse_ollama_section_plan(payload)
    if parsed.valid:
        return parsed
    fallback_target_duration = 30.0
    fallback_difficulty = "normal"
    fallback_style = "modern_glow"
    fallback_sync = "medium"
    if isinstance(parsed.json_payload, dict):
        maybe_level = parsed.json_payload.get("level_plan")
        if isinstance(maybe_level, dict):
            try:
                fallback_target_duration = float(maybe_level.get("target_duration", fallback_target_duration))
            except (TypeError, ValueError):
                fallback_target_duration = 30.0
            fallback_difficulty = str(maybe_level.get("difficulty", fallback_difficulty)).lower()
            fallback_style = str(maybe_level.get("style", fallback_style))
            fallback_sync = str(maybe_level.get("sync_intensity", fallback_sync)).lower()
    fallback = build_template_level_plan(
        prompt=prompt,
        level_name=fallback_level_name,
        object_budget=object_budget,
        target_duration=fallback_target_duration,
        difficulty=fallback_difficulty,
        style=fallback_style,
        sync_intensity=fallback_sync,
    )
    res = PlannerParseResult(
        plan=fallback,
        errors=list(parsed.errors),
        fallback_used=True,
        fallback_reason="invalid_ollama_planner_output",
        raw_payload=parsed.raw_payload,
        json_payload=parsed.json_payload,
        forbidden_fields=list(parsed.forbidden_fields),
        forbidden_field_paths=list(parsed.forbidden_field_paths),
        schema_error_path=parsed.schema_error_path,
        missing_required_fields=list(parsed.missing_required_fields),
        empty_required_fields=list(parsed.empty_required_fields),
        wrong_location_fields=list(parsed.wrong_location_fields),
        schema_error_message=parsed.schema_error_message,
        planner_failure_stage=parsed.planner_failure_stage,
        planner_failure_reason_detail=parsed.planner_failure_reason_detail,
        normalized_shape_repairs=list(parsed.normalized_shape_repairs),
        repair_prompt_sent=parsed.repair_prompt_sent,
        repair_response_preview=parsed.repair_response_preview,
        repair_success=parsed.repair_success,
        planner_repair_skipped_reason=parsed.planner_repair_skipped_reason,
    )
    return res


def _result_with_diagnostics(
    *,
    errors: list[str],
    raw_payload: str | None,
    json_payload: dict[str, Any] | None,
    missing_required_fields: list[str] | None = None,
    empty_required_fields: list[str] | None = None,
    wrong_location_fields: list[str] | None = None,
    normalized_shape_repairs: list[str] | None = None,
    planner_failure_stage: str | None = None,
    planner_failure_reason_detail: str | None = None,
) -> PlannerParseResult:
    forbidden_paths = _forbidden_field_paths_from_errors(errors)
    return PlannerParseResult(
        errors=errors,
        raw_payload=raw_payload,
        json_payload=json_payload,
        forbidden_fields=_field_names_from_paths(forbidden_paths),
        forbidden_field_paths=forbidden_paths,
        schema_error_path=_schema_error_path_from_errors(errors),
        missing_required_fields=list(missing_required_fields or []),
        empty_required_fields=list(empty_required_fields or []),
        wrong_location_fields=list(wrong_location_fields or []),
        schema_error_message=_schema_error_message(errors, missing_required_fields, empty_required_fields),
        planner_failure_stage=planner_failure_stage or ("schema_validation" if errors else None),
        planner_failure_reason_detail=planner_failure_reason_detail or "",
        normalized_shape_repairs=list(normalized_shape_repairs or []),
    )


def _forbidden_field_paths_from_errors(errors: list[str]) -> list[str]:
    paths: list[str] = []
    for error in errors:
        if "forbidden_planner_field" not in error:
            continue
        path = error.split(":", 1)[0]
        if path and path not in paths:
            paths.append(path)
    return paths


def _field_names_from_paths(paths: list[str]) -> list[str]:
    fields: list[str] = []
    for path in paths:
        field = re.split(r"\.|\[", path)[-1].rstrip("]")
        if field and field not in fields:
            fields.append(field)
    return fields


def _schema_error_path_from_errors(errors: list[str]) -> str | None:
    for error in errors:
        if ":" in error:
            return error.split(":", 1)[0]
    return "$" if errors else None


def _append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def _collect_level_plan_missing_paths(errors: list[str], missing_required_fields: list[str]) -> None:
    for error in errors:
        if not error.startswith("level_plan_missing:"):
            continue
        fields = error.split(":", 1)[1]
        for field in fields.split(","):
            field_name = field.strip()
            if field_name:
                _append_unique(missing_required_fields, f"$.level_plan.{field_name}")


def _collect_section_missing_paths(errors: list[str], missing_required_fields: list[str]) -> None:
    for error in errors:
        if ":missing:" not in error:
            continue
        prefix, fields = error.split(":missing:", 1)
        for field in fields.split(","):
            field_name = field.strip()
            if field_name:
                _append_unique(missing_required_fields, f"$.{prefix}.{field_name}")


def _schema_error_message(
    errors: list[str],
    missing_required_fields: list[str] | None,
    empty_required_fields: list[str] | None,
) -> str | None:
    parts = []
    missing = missing_required_fields or []
    empty = empty_required_fields or []
    
    # Check for critical missing top-level components first
    if "$.level_plan" in missing and "$.sections" in missing:
        return "Both top-level level_plan and non-empty top-level sections are required."
    
    if "$.level_plan" in missing:
        parts.append("level_plan is missing")
    if "$.sections" in missing:
        parts.append("sections is missing")
    elif "$.sections" in empty:
        parts.append("sections must be a non-empty array")
    
    if parts:
        return ". ".join(parts).capitalize() + "."

    if errors:
        # Filter and prioritize errors
        relevant = [e for e in errors if not e.startswith("unknown_top_level_keys")]
        if not relevant:
            relevant = errors
        return "; ".join(relevant[:4])
    return None


def _planner_failure_reason_detail(
    *,
    missing_required_fields: list[str],
    empty_required_fields: list[str],
    wrong_location_fields: list[str],
) -> str:
    parts: list[str] = []
    if missing_required_fields:
        parts.append(f"missing={','.join(missing_required_fields)}")
    if wrong_location_fields:
        parts.append(f"wrong_location={','.join(wrong_location_fields)}")
    if empty_required_fields:
        parts.append(f"empty={','.join(empty_required_fields)}")
    return " ".join(parts)


def build_template_level_plan(
    *,
    prompt: str = "",
    level_name: str = "fallback_plan",
    object_budget: int = 500,
    target_duration: float = 30.0,
    difficulty: str = "normal",
    style: str = "modern_glow",
    sync_intensity: str = "medium",
) -> LevelPlan:
    notes = prompt.strip()[:180] or "deterministic fallback section plan"
    sections: list[SectionPlan]
    if target_duration >= 190.0:
        sections = [
            SectionPlan("s001", 0.0, 20.0, "cube", "1x", 0.20, "simple_cube_intro", ["block", "spike", "orb", "pad"], ["glow_spam", "camera_shake"], 0, [], [], "fallback intro"),
            SectionPlan("s002", 20.0, 45.0, "cube", "1x", 0.28, "cube_progression", ["block", "spike", "orb", "pad"], ["glow_spam", "camera_shake"], 0, [], [], "fallback build-up"),
            SectionPlan("s003", 45.0, 70.0, "ship", "1x", 0.24, "ship_straight_fly", ["block", "spike", "orb", "pad"], ["glow_spam", "camera_shake"], 0, [], [], "fallback ship section"),
            SectionPlan("s004", 70.0, 95.0, "cube", "1x", 0.35, "cube_sync_motif", ["block", "spike", "orb", "pad"], ["glow_spam", "camera_shake"], 0, [], [], "fallback rhythm section"),
            SectionPlan("s005", 95.0, 120.0, "ball", "1x", 0.30, "ball_switch_intro", ["block", "spike", "orb", "pad"], ["glow_spam", "camera_shake"], 0, [], [], "fallback ball section"),
            SectionPlan("s006", 120.0, 145.0, "cube", "1x", 0.36, "cube_midgame", ["block", "spike", "orb", "pad"], ["glow_spam", "camera_shake"], 0, [], [], "fallback midgame"),
            SectionPlan("s007", 145.0, 170.0, "cube", "2x", 0.42, "cube_speedup", ["block", "spike", "orb", "pad"], ["glow_spam", "camera_shake"], 0, [], [], "fallback speedup"),
            SectionPlan("s008", 170.0, 198.0, "cube", "1x", 0.24, "cube_outro", ["block", "spike", "orb", "pad"], ["glow_spam", "camera_shake"], 0, [], [], "fallback outro"),
        ]
    else:
        sections = [
            SectionPlan(
                section_id="s001",
                time_start=0.0,
                time_end=min(max(8.0, target_duration), target_duration if target_duration > 0 else 8.0),
                game_mode="cube",
                speed="1x",
                density=0.35,
                primary_pattern="intro_platforming",
                allowed_object_families=["block", "spike", "orb", "pad"],
                forbidden_features=["unbounded_trigger_spam", "raw_gmd_generation"],
                trigger_budget=3,
                group_symbols=[GroupSymbol("intro_blocks")],
                color_symbols=[ColorSymbol("accent_primary")],
                design_notes=notes,
            ),
        ]
    return LevelPlan(
        level_name=level_name,
        difficulty=str(difficulty or "normal").lower(),
        target_duration=max(1.0, float(target_duration)),
        object_budget=max(1, int(object_budget)),
        style=str(style or "modern_glow"),
        sync_intensity=str(sync_intensity or "medium").lower(),
        sections=sections,
    )


def _validate_level_plan_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "level_name",
        "difficulty",
        "target_duration",
        "object_budget",
        "style",
        "sync_intensity",
    }
    missing = required - set(payload)
    if missing:
        errors.append(f"level_plan_missing:{','.join(sorted(missing))}")
    if str(payload.get("difficulty", "")).lower() not in DIFFICULTIES:
        errors.append("level_plan_unknown_difficulty")
    if str(payload.get("sync_intensity", "")).lower() not in SYNC_INTENSITIES:
        errors.append("level_plan_unknown_sync_intensity")
    
    td_raw = payload.get("target_duration")
    if td_raw is not None:
        try:
            target_duration = float(td_raw)
            if target_duration <= 0 or target_duration > 600:
                errors.append("level_plan_target_duration_out_of_range")
        except (TypeError, ValueError):
            errors.append("level_plan_target_duration_invalid")
    else:
        errors.append("level_plan_target_duration_invalid")

    ob_raw = payload.get("object_budget")
    if ob_raw is not None:
        try:
            object_budget = int(ob_raw)
            if object_budget <= 0 or object_budget > 40000:
                errors.append("level_plan_object_budget_out_of_range")
        except (TypeError, ValueError):
            errors.append("level_plan_object_budget_invalid")
    else:
        errors.append("level_plan_object_budget_invalid")
    return errors


def _parse_section_payload(payload: dict[str, Any], index: int) -> tuple[SectionPlan | None, list[str]]:
    prefix = f"sections[{index}]"
    errors: list[str] = []
    
    # Required fields that must be present
    required = {
        "section_id",
        "time_start",
        "time_end",
        "game_mode",
        "speed",
        "density",
        "primary_pattern",
    }
    
    missing = required - set(payload)
    if missing:
        errors.append(f"{prefix}:missing:{','.join(sorted(missing))}")
    
    # Defaults for optional fields
    if "allowed_object_families" not in payload:
        payload["allowed_object_families"] = ["block", "spike", "orb", "pad"]
    if "forbidden_features" not in payload:
        payload["forbidden_features"] = []
    if "trigger_budget" not in payload:
        payload["trigger_budget"] = 0
    if "group_symbols" not in payload:
        payload["group_symbols"] = []
    if "design_notes" not in payload:
        payload["design_notes"] = ""

    game_mode = str(payload.get("game_mode", "")).lower()
    speed = str(payload.get("speed", "")).lower()
    if game_mode not in GAME_MODES:
        errors.append(f"{prefix}:unknown_game_mode")
    if speed not in SPEEDS:
        errors.append(f"{prefix}:unknown_speed")
    
    ts_raw = payload.get("time_start")
    te_raw = payload.get("time_end")
    if ts_raw is not None and te_raw is not None:
        try:
            time_start = float(ts_raw)
            time_end = float(te_raw)
            if time_start < 0 or time_end <= time_start:
                errors.append(f"{prefix}:time_range_invalid")
        except (TypeError, ValueError):
            time_start = 0.0
            time_end = 0.0
            errors.append(f"{prefix}:time_range_invalid")
    else:
        time_start = 0.0
        time_end = 0.0
        errors.append(f"{prefix}:time_range_invalid")

    d_raw = payload.get("density")
    if d_raw is not None:
        try:
            density = float(d_raw)
            if density < 0.0 or density > 1.0:
                errors.append(f"{prefix}:density_out_of_range")
        except (TypeError, ValueError):
            density = 0.0
            errors.append(f"{prefix}:density_invalid")
    else:
        density = 0.0
        errors.append(f"{prefix}:density_invalid")

    tb_raw = payload.get("trigger_budget")
    if tb_raw is not None:
        try:
            trigger_budget = int(tb_raw)
            if trigger_budget < 0 or trigger_budget > 128:
                errors.append(f"{prefix}:trigger_budget_out_of_range")
        except (TypeError, ValueError):
            trigger_budget = 0
            errors.append(f"{prefix}:trigger_budget_invalid")
    else:
        trigger_budget = 0

    allowed = payload.get("allowed_object_families")
    forbidden = payload.get("forbidden_features")
    groups = payload.get("group_symbols")
    colors = payload.get("color_symbols", [])
    
    if not isinstance(allowed, list):
        errors.append(f"{prefix}:allowed_object_families_must_be_list")
    if not isinstance(forbidden, list):
        errors.append(f"{prefix}:forbidden_features_must_be_list")
    if not isinstance(groups, list):
        errors.append(f"{prefix}:group_symbols_must_be_list")
    if colors is not None and not isinstance(colors, list):
        errors.append(f"{prefix}:color_symbols_must_be_list")

    if errors:
        return None, errors
        
    return (
        SectionPlan(
            section_id=str(payload["section_id"]),
            time_start=time_start,
            time_end=time_end,
            game_mode=game_mode,
            speed=speed,
            density=density,
            primary_pattern=str(payload["primary_pattern"]),
            allowed_object_families=[str(item) for item in (allowed or []) if item],
            forbidden_features=[str(item) for item in (forbidden or []) if item],
            trigger_budget=trigger_budget,
            group_symbols=[GroupSymbol(str(item)) for item in (groups or []) if item],
            color_symbols=[ColorSymbol(str(item)) for item in (colors or []) if item],
            design_notes=str(payload["design_notes"]),
        ),
        [],
    )


def _string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, (str, int, float)) for item in value)


def _reject_forbidden_shape(value: Any, *, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, str):
        if _looks_like_raw_gmd(value):
            errors.append(f"{path}:raw_gmd_output_rejected")
        return errors
    if isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_reject_forbidden_shape(item, path=f"{path}[{index}]"))
        return errors
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_KEYS:
                errors.append(f"{path}.{key_text}:forbidden_planner_field")
            errors.extend(_reject_forbidden_shape(item, path=f"{path}.{key_text}"))
    return errors


def _looks_like_raw_gmd(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if RAW_GMD_PATTERN.search(stripped):
        return True
    parts = stripped.split(",")
    return len(parts) >= 8 and parts[0].isdigit() and any(";" in item for item in parts)


def _normalize_dict_aliases(data: dict[str, Any]) -> None:
    if "level_plan" in data and isinstance(data["level_plan"], dict):
        lp = data["level_plan"]
        
        # Target duration normalization
        if "duration" in lp and "target_duration" not in lp:
            lp["target_duration"] = lp.pop("duration")
        if "name" in lp and "level_name" not in lp:
            lp["level_name"] = lp.pop("name")

        if "difficulty" in lp:
            diff = str(lp["difficulty"]).lower()
            diff = diff.replace(" gameplay", "").replace(" mode", "").replace(" difficulty", "")
            if "-" in diff:
                parts = diff.split("-")
                for p in reversed(parts):
                    if p in DIFFICULTIES:
                        diff = p
                        break
            lp["difficulty"] = diff

        if "sync_intensity" in lp:
            sync = str(lp["sync_intensity"]).lower()
            if sync in ["moderate", "medium-high"]:
                lp["sync_intensity"] = "medium"
            elif sync in ["intense", "extreme", "very high"]:
                lp["sync_intensity"] = "high"
            elif sync in ["calm", "relaxed", "very low"]:
                lp["sync_intensity"] = "low"

    if "sections" in data and isinstance(data["sections"], list):
        for section in data["sections"]:
            if not isinstance(section, dict):
                continue
            
            # Key aliases
            if "target_density" in section and "density" not in section:
                section["density"] = section.pop("target_density")
            if "allowed_objects" in section and "allowed_object_families" not in section:
                section["allowed_object_families"] = section.pop("allowed_objects")
            if "forbidden" in section and "forbidden_features" not in section:
                section["forbidden_features"] = section.pop("forbidden")
            if "notes" in section and "design_notes" not in section:
                section["design_notes"] = section.pop("notes")
            if "design_note" in section and "design_notes" not in section:
                section["design_notes"] = section.pop("design_note")
            if "group_names" in section and "group_symbols" not in section:
                section["group_symbols"] = section.pop("group_names")
            
            # Value normalization
            if "game_mode" in section:
                gm = str(section["game_mode"]).lower()
                gm = gm.replace(" gameplay", "").replace(" mode", "").replace(" session", "")
                if "cube" in gm: section["game_mode"] = "cube"
                elif "ship" in gm: section["game_mode"] = "ship"
                elif "ball" in gm: section["game_mode"] = "ball"
                elif "ufo" in gm: section["game_mode"] = "ufo"
                elif "wave" in gm: section["game_mode"] = "wave"
                elif "robot" in gm: section["game_mode"] = "robot"
                elif "spider" in gm: section["game_mode"] = "spider"
                else: section["game_mode"] = gm
            
            if "speed" in section:
                speed_str = str(section["speed"]).lower()
                speed_str = speed_str.replace(" speed", "").replace("x", "").strip()
                if speed_str in ["half", "0.5", "slow", "0.5x"]:
                    section["speed"] = "0.5x"
                elif speed_str in ["normal", "1", "1.0", "1x"]:
                    section["speed"] = "1x"
                elif speed_str in ["fast", "2", "2.0", "2x"]:
                    section["speed"] = "2x"
                elif speed_str in ["very fast", "3", "3.0", "3x"]:
                    section["speed"] = "3x"
                elif speed_str in ["super fast", "4", "4.0", "4x"]:
                    section["speed"] = "4x"
                else:
                    if re.match(r"^\d(\.\d)?$", speed_str):
                        section["speed"] = f"{speed_str}x"
                    else:
                        section["speed"] = "1x" # fallback for garbage
