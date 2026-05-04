# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

from gmdgen.gd.plans import ObjectPlan, SectionPlan, TriggerMode, TriggerPlan
from gmdgen.gd.triggers import (
    apply_trigger_property_defaults,
    get_trigger_schema,
    list_supported_triggers,
    validate_trigger_properties,
)
from gmdgen.ai.normalization import (
    AIPlanNormalizationReport,
    allowed_object_roles,
    normalize_ai_level_plan_response_payload,
)

_ALLOWED_RESPONSE_KEYS = {
    "sections",
    "gameplay_events",
    "object_plans",
    "trigger_plans",
    "speed_plan",
    "reasoning_summary",
    "safety_notes",
    "expected_sync_notes",
    "metadata",
}
_ALLOWED_OBJECT_KEYS = {
    "object_id",
    "x",
    "y",
    "role",
    "group_ids",
    "color_channel",
    "editor_layer",
    "z_layer",
    "z_order",
    "rotation",
    "scale",
    "safety_flags",
    "beat_aligned_time",
    "sync_error",
}
_ALLOWED_TRIGGER_KEYS = {
    "trigger_type",
    "object_id",
    "x",
    "y",
    "target_group",
    "secondary_group",
    "duration",
    "easing",
    "spawn_delay",
    "multi_trigger",
    "editor_disable",
    "beat_aligned_time",
    "sync_error",
    "properties",
}
_ALLOWED_ROLES = set(allowed_object_roles())


AI_LEVEL_PLAN_JSON_SCHEMA: dict[str, Any] = {
    "name": "gmdgen_ai_level_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "sections",
            "gameplay_events",
            "object_plans",
            "trigger_plans",
            "speed_plan",
            "reasoning_summary",
            "safety_notes",
            "expected_sync_notes",
            "metadata",
        ],
        "properties": {
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "section_id",
                        "start_time",
                        "end_time",
                        "start_x",
                        "end_x",
                        "section_type",
                        "gameplay_mode",
                        "speed_state",
                        "density_target",
                        "decoration_intensity",
                        "trigger_intensity",
                        "difficulty_target",
                    ],
                    "properties": {
                        "section_id": {"type": "integer"},
                        "start_time": {"type": "number"},
                        "end_time": {"type": "number"},
                        "start_x": {"type": "number"},
                        "end_x": {"type": "number"},
                        "section_type": {"type": "string"},
                        "gameplay_mode": {"type": "string"},
                        "speed_state": {"type": "string"},
                        "density_target": {"type": "number"},
                        "decoration_intensity": {"type": "number"},
                        "trigger_intensity": {"type": "number"},
                        "difficulty_target": {"type": "number"},
                    },
                },
            },
            "gameplay_events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "time",
                        "x",
                        "event_type",
                        "importance",
                        "beat_index",
                        "sync_error",
                        "section_id",
                    ],
                    "properties": {
                        "time": {"type": "number"},
                        "x": {"type": "number"},
                        "event_type": {"type": "string"},
                        "importance": {"type": "number"},
                        "beat_index": {"type": "integer"},
                        "sync_error": {"type": "number"},
                        "section_id": {"type": "integer"},
                    },
                },
            },
            "object_plans": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "object_id",
                        "x",
                        "y",
                        "role",
                        "group_ids",
                        "color_channel",
                        "editor_layer",
                        "z_layer",
                        "z_order",
                        "rotation",
                        "scale",
                        "safety_flags",
                        "beat_aligned_time",
                        "sync_error",
                    ],
                    "properties": {
                        "object_id": {"type": ["integer", "string"]},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "role": {"type": "string", "enum": list(allowed_object_roles())},
                        "group_ids": {"type": "array", "items": {"type": "integer"}},
                        "color_channel": {"type": ["integer", "null"]},
                        "editor_layer": {"type": ["integer", "null"]},
                        "z_layer": {"type": ["integer", "null"]},
                        "z_order": {"type": ["integer", "null"]},
                        "rotation": {"type": "number"},
                        "scale": {"type": "number"},
                        "safety_flags": {
                            "type": ["object", "null"],
                            "additionalProperties": False,
                            "required": [
                                "importance",
                                "section_id",
                                "source",
                                "note",
                            ],
                            "properties": {
                                "importance": {"type": ["number", "null"]},
                                "section_id": {"type": ["integer", "null"]},
                                "source": {"type": ["string", "null"]},
                                "note": {"type": ["string", "null"]},
                            },
                        },
                        "beat_aligned_time": {"type": ["number", "null"]},
                        "sync_error": {"type": "number"},
                    },
                },
            },
            "trigger_plans": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "trigger_type",
                        "object_id",
                        "x",
                        "y",
                        "target_group",
                        "secondary_group",
                        "duration",
                        "easing",
                        "spawn_delay",
                        "multi_trigger",
                        "editor_disable",
                        "beat_aligned_time",
                        "sync_error",
                        "properties",
                    ],
                    "properties": {
                        "trigger_type": {"type": "string"},
                        "object_id": {"type": ["integer", "string", "null"]},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "target_group": {"type": ["integer", "null"]},
                        "secondary_group": {"type": ["integer", "null"]},
                        "duration": {"type": ["number", "null"]},
                        "easing": {"type": ["string", "null"]},
                        "spawn_delay": {"type": ["number", "null"]},
                        "multi_trigger": {"type": "boolean"},
                        "editor_disable": {"type": "boolean"},
                        "beat_aligned_time": {"type": ["number", "null"]},
                        "sync_error": {"type": "number"},
                        "properties": {
                            "type": ["object", "null"],
                            "additionalProperties": False,
                            "required": [
                                "move_x",
                                "move_y",
                                "opacity",
                                "color_channel",
                                "copy_color_channel",
                                "exclusive",
                                "trigger_kind",
                                "purpose",
                                "target_role",
                                "intensity",
                                "duration_hint",
                                "section_id",
                            ],
                            "properties": {
                                "move_x": {"type": ["number", "null"]},
                                "move_y": {"type": ["number", "null"]},
                                "opacity": {"type": ["number", "null"]},
                                "color_channel": {"type": ["integer", "null"]},
                                "copy_color_channel": {"type": ["integer", "null"]},
                                "exclusive": {"type": ["boolean", "null"]},
                                "trigger_kind": {"type": ["string", "null"]},
                                "purpose": {"type": ["string", "null"]},
                                "target_role": {"type": ["string", "null"]},
                                "intensity": {"type": ["number", "null"]},
                                "duration_hint": {"type": ["number", "null"]},
                                "section_id": {"type": ["integer", "null"]},
                            },
                        },
                    },
                },
            },
            "speed_plan": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "time",
                        "x",
                        "speed_state",
                        "object_id",
                    ],
                    "properties": {
                        "time": {"type": "number"},
                        "x": {"type": "number"},
                        "speed_state": {"type": "string"},
                        "object_id": {"type": ["string", "null"]},
                    },
                },
            },
            "reasoning_summary": {"type": "string"},
            "safety_notes": {"type": "array", "items": {"type": "string"}},
            "expected_sync_notes": {"type": "array", "items": {"type": "string"}},
            "metadata": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "required": [
                    "debug_artifact_path",
                    "fallback_reason",
                    "provider_note",
                ],
                "properties": {
                    "debug_artifact_path": {"type": ["string", "null"]},
                    "fallback_reason": {"type": ["string", "null"]},
                    "provider_note": {"type": ["string", "null"]},
                },
            },
        },
    },
}

def build_ai_level_plan_schema() -> dict[str, Any]:
    return deepcopy(AI_LEVEL_PLAN_JSON_SCHEMA["schema"])

def validate_ai_plan_response_locally(
    response: AILevelPlanResponse,
    *,
    object_budget: int,
    max_group_id: int,
    safe_mode: bool,
    section_plans: list[SectionPlan] | None = None,
) -> list[str]:
    return validate_ai_level_plan_response_schema(
        response,
        object_budget=object_budget,
        max_group_id=max_group_id,
        safe_mode=safe_mode,
        section_plans=section_plans,
    )

def _schema_allows_object(node: dict[str, Any]) -> bool:
    schema_type = node.get("type")
    if schema_type == "object":
        return True
    if isinstance(schema_type, list) and "object" in schema_type:
        return True
    return False

@dataclass(slots=True)
class AILevelPlanRequest:
    project_goal: str
    generation_mode: str
    difficulty: str | float
    safe_mode: bool
    object_budget: int
    song_offset: float
    start_speed: str
    audio_summary: dict[str, Any]
    beat_summary: dict[str, Any]
    onset_summary: dict[str, Any]
    section_plans: list[dict[str, Any]]
    time_x_summary: dict[str, Any]
    trigger_schema_summary: dict[str, Any]
    playability_rules_summary: dict[str, Any]
    user_prompt: str = ""
    style_reference_summary: dict[str, Any] = field(default_factory=dict)
    learned_style_summary: dict[str, Any] = field(default_factory=dict)
    retrieved_motifs: list[dict[str, Any]] = field(default_factory=list)
    learned_object_distribution: dict[str, Any] = field(default_factory=dict)
    learned_trigger_distribution: dict[str, Any] = field(default_factory=dict)
    learned_density_profile: dict[str, Any] = field(default_factory=dict)
    learned_failure_patterns: list[str] = field(default_factory=list)
    learned_success_patterns: list[str] = field(default_factory=list)
    retrieved_context: list[dict[str, Any]] = field(default_factory=list)
    output_requirements: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(slots=True)
class AILevelPlanResponse:
    sections: list[dict[str, Any]] = field(default_factory=list)
    gameplay_events: list[dict[str, Any]] = field(default_factory=list)
    object_plans: list[dict[str, Any]] = field(default_factory=list)
    trigger_plans: list[dict[str, Any]] = field(default_factory=list)
    speed_plan: list[dict[str, Any]] = field(default_factory=list)
    reasoning_summary: str = ""
    safety_notes: list[str] = field(default_factory=list)
    expected_sync_notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_response: str = ""
    provider: str = "unknown"
    model: str = ""
    fallback_used: bool = False
    fallback_reason: str = ""
    valid: bool = True
    validation_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(slots=True)
class AIPlanConversionResult:
    response: AILevelPlanResponse | None
    object_plans: list[ObjectPlan] = field(default_factory=list)
    trigger_plans: list[TriggerPlan] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    normalization_report: AIPlanNormalizationReport = field(default_factory=AIPlanNormalizationReport)

    @property
    def valid(self) -> bool:
        return self.response is not None and not self.errors

def parse_ai_level_plan_response(raw_response: str | bytes | dict[str, Any] | AILevelPlanResponse) -> AILevelPlanResponse:
    if isinstance(raw_response, AILevelPlanResponse):
        return raw_response
    if isinstance(raw_response, bytes):
        raw_response = raw_response.decode("utf-8", errors="replace")
    if isinstance(raw_response, str):
        payload = json.loads(_extract_json_text(raw_response))
        raw_text = raw_response
    elif isinstance(raw_response, dict):
        payload = raw_response
        raw_text = json.dumps(raw_response, ensure_ascii=False)
    else:
        raise TypeError(f"Unsupported AI response type: {type(raw_response)!r}")
    if not isinstance(payload, dict):
        raise ValueError("AI level plan response must be a JSON object")
    unknown = sorted(set(payload) - _ALLOWED_RESPONSE_KEYS)
    if unknown:
        raise ValueError(f"unknown_ai_response_keys: {unknown}")
    return AILevelPlanResponse(
        sections=_list_of_dicts(payload.get("sections", []), "sections"),
        gameplay_events=_list_of_dicts(payload.get("gameplay_events", []), "gameplay_events"),
        object_plans=_list_of_dicts(payload.get("object_plans", []), "object_plans"),
        trigger_plans=_list_of_dicts(payload.get("trigger_plans", []), "trigger_plans"),
        speed_plan=_list_of_dicts(payload.get("speed_plan", []), "speed_plan"),
        reasoning_summary=str(payload.get("reasoning_summary", ""))[:1200],
        safety_notes=[str(item)[:300] for item in payload.get("safety_notes", []) if isinstance(item, (str, int, float))],
        expected_sync_notes=[str(item)[:300] for item in payload.get("expected_sync_notes", []) if isinstance(item, (str, int, float))],
        metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {},
        raw_response=raw_text,
    )

def validate_ai_level_plan_response_schema(
    response: AILevelPlanResponse,
    *,
    object_budget: int,
    max_group_id: int,
    safe_mode: bool,
    section_plans: list[SectionPlan] | None = None,
) -> list[str]:
    errors: list[str] = []
    if len(response.object_plans) + len(response.trigger_plans) > object_budget:
        errors.append("ai_output_object_budget_exceeded")
    allowed_triggers = set(list_supported_triggers(TriggerMode.SAFE if safe_mode else TriggerMode.ADVANCED))
    section_min_x = min((section.start_x for section in section_plans or []), default=-1e9)
    section_max_x = max((section.end_x for section in section_plans or []), default=1e9)

    for idx, plan in enumerate(response.object_plans):
        unknown = sorted(set(plan) - _ALLOWED_OBJECT_KEYS)
        if unknown:
            errors.append(f"unknown_object_plan_keys[{idx}]: {unknown}")
        for field_name in ("object_id", "x", "y", "role"):
            if field_name not in plan:
                errors.append(f"missing_object_plan_field[{idx}]: {field_name}")
        if not _finite(plan.get("x")) or not _finite(plan.get("y")):
            errors.append(f"nonfinite_object_coordinate[{idx}]")
        if not _positive_intish(plan.get("object_id")):
            errors.append(f"invalid_object_id[{idx}]")
        role = str(plan.get("role", ""))
        if role not in _ALLOWED_ROLES:
            errors.append(f"unsupported_object_role[{idx}]: {role}")
        x_value = _float_or_none(plan.get("x"))
        if x_value is not None and not (section_min_x - 240 <= x_value <= section_max_x + 240):
            errors.append(f"object_outside_section_bounds[{idx}]")
        group_ids = plan.get("group_ids", [])
        if group_ids is not None and not isinstance(group_ids, list):
            errors.append(f"invalid_group_ids_type[{idx}]")
        for group_id in group_ids or []:
            if not isinstance(group_id, int) or not (1 <= group_id <= max_group_id):
                errors.append(f"group_id_bounds[{idx}]: {group_id}")

    for idx, plan in enumerate(response.trigger_plans):
        unknown = sorted(set(plan) - _ALLOWED_TRIGGER_KEYS)
        if unknown:
            errors.append(f"unknown_trigger_plan_keys[{idx}]: {unknown}")
        trigger_type = str(plan.get("trigger_type", "")).lower()
        if trigger_type not in allowed_triggers:
            errors.append(f"unsupported_trigger[{idx}]: {trigger_type}")
            continue
        if not _finite(plan.get("x")) or not _finite(plan.get("y")):
            errors.append(f"nonfinite_trigger_coordinate[{idx}]")
        target = plan.get("target_group")
        if target is not None and (not isinstance(target, int) or not (1 <= target <= max_group_id)):
            errors.append(f"trigger_group_bounds[{idx}]: {target}")
        schema = get_trigger_schema(trigger_type)
        object_id = str(plan.get("object_id") or (schema.object_id if schema else ""))
        trigger_plan = TriggerPlan(
            trigger_type=trigger_type,
            object_id=object_id,
            x=float(plan.get("x", 0.0)),
            y=float(plan.get("y", 0.0)),
            target_group=target,
            secondary_group=plan.get("secondary_group"),
            duration=float(plan.get("duration") or 0.0),
            easing=plan.get("easing"),
            spawn_delay=float(plan.get("spawn_delay") or 0.0),
            multi_trigger=bool(plan.get("multi_trigger", False)),
            editor_disable=bool(plan.get("editor_disable", False)),
            beat_aligned_time=plan.get("beat_aligned_time"),
            sync_error=float(plan.get("sync_error", 0.0)),
            properties=dict(plan.get("properties", {})) if isinstance(plan.get("properties", {}), dict) else {},
        )
        defaulted = apply_trigger_property_defaults(
            trigger_plan,
            TriggerMode.SAFE if safe_mode else TriggerMode.ADVANCED,
        )
        if defaulted is None:
            errors.append(f"trigger_defaults_rejected[{idx}]: {trigger_type}")
            continue
        issues = validate_trigger_properties(
            defaulted,
            TriggerMode.SAFE if safe_mode else TriggerMode.ADVANCED,
            max_group_id=max_group_id,
        )
        errors.extend(f"trigger_schema[{idx}]: {issue}" for issue in issues)
    return errors

def convert_ai_response_to_plans(
    response: AILevelPlanResponse,
    *,
    object_budget: int,
    max_group_id: int,
    safe_mode: bool,
    section_plans: list[SectionPlan] | None = None,
) -> AIPlanConversionResult:
    normalization_report = _normalize_response_in_place(
        response,
        safe_mode=safe_mode,
        max_group_id=max_group_id,
        section_plans=section_plans,
    )
    errors = validate_ai_level_plan_response_schema(
        response,
        object_budget=object_budget,
        max_group_id=max_group_id,
        safe_mode=safe_mode,
        section_plans=section_plans,
    )
    if errors:
        response.valid = False
        response.validation_errors = list(errors)
        return AIPlanConversionResult(
            response=response,
            errors=errors,
            warnings=list(normalization_report.warnings),
            normalization_report=normalization_report,
        )

    object_plans: list[ObjectPlan] = []
    trigger_plans: list[TriggerPlan] = []
    for raw in response.object_plans:
        object_plans.append(
            ObjectPlan(
                object_id=str(raw["object_id"]),
                x=float(raw["x"]),
                y=float(raw["y"]),
                role=str(raw.get("role", "ai_structure")),
                group_ids=[int(group_id) for group_id in raw.get("group_ids", [])],
                color_channel=_optional_int(raw.get("color_channel")),
                editor_layer=_optional_int(raw.get("editor_layer")),
                z_layer=_optional_int(raw.get("z_layer")),
                z_order=_optional_int(raw.get("z_order")),
                rotation=float(raw.get("rotation", 0.0)),
                scale=max(0.1, min(4.0, float(raw.get("scale", 1.0)))),
                safety_flags=dict(raw.get("safety_flags", {})) if isinstance(raw.get("safety_flags", {}), dict) else {},
                beat_aligned_time=_optional_float(raw.get("beat_aligned_time")),
                sync_error=float(raw.get("sync_error", 0.0)),
            )
        )

    trigger_mode = TriggerMode.SAFE if safe_mode else TriggerMode.ADVANCED
    for raw in response.trigger_plans:
        trigger_type = str(raw["trigger_type"]).lower()
        schema = get_trigger_schema(trigger_type)
        trigger = TriggerPlan(
            trigger_type=trigger_type,
            object_id=str(raw.get("object_id") or (schema.object_id if schema else "")),
            x=float(raw["x"]),
            y=float(raw["y"]),
            target_group=_optional_int(raw.get("target_group")),
            secondary_group=_optional_int(raw.get("secondary_group")),
            duration=float(raw.get("duration") or 0.0),
            easing=raw.get("easing"),
            spawn_delay=float(raw.get("spawn_delay") or 0.0),
            multi_trigger=bool(raw.get("multi_trigger", False)),
            editor_disable=bool(raw.get("editor_disable", False)),
            beat_aligned_time=_optional_float(raw.get("beat_aligned_time")),
            sync_error=float(raw.get("sync_error", 0.0)),
            properties=dict(raw.get("properties", {})) if isinstance(raw.get("properties", {}), dict) else {},
        )
        defaulted = apply_trigger_property_defaults(trigger, trigger_mode)
        if defaulted is not None:
            trigger_plans.append(defaulted)
    return AIPlanConversionResult(
        response=response,
        object_plans=object_plans,
        trigger_plans=trigger_plans,
        warnings=list(normalization_report.warnings),
        normalization_report=normalization_report,
    )

def reject_or_repair_invalid_ai_output(
    raw_response: str | bytes | dict[str, Any] | AILevelPlanResponse,
    *,
    object_budget: int,
    max_group_id: int,
    safe_mode: bool,
    section_plans: list[SectionPlan] | None = None,
) -> AIPlanConversionResult:
    try:
        response = parse_ai_level_plan_response(raw_response)
    except Exception as exc:  # noqa: BLE001
        return AIPlanConversionResult(response=None, errors=[f"ai_response_parse_failed: {exc}"])
    return convert_ai_response_to_plans(
        response,
        object_budget=object_budget,
        max_group_id=max_group_id,
        safe_mode=safe_mode,
        section_plans=section_plans,
    )

def _normalize_response_in_place(
    response: AILevelPlanResponse,
    *,
    safe_mode: bool,
    max_group_id: int = 9999,
    section_plans: list[SectionPlan] | None = None,
) -> AIPlanNormalizationReport:
    payload = {
        "sections": response.sections,
        "gameplay_events": response.gameplay_events,
        "object_plans": response.object_plans,
        "trigger_plans": response.trigger_plans,
        "speed_plan": response.speed_plan,
        "reasoning_summary": response.reasoning_summary,
        "safety_notes": response.safety_notes,
        "expected_sync_notes": response.expected_sync_notes,
        "metadata": response.metadata,
    }
    normalized, report = normalize_ai_level_plan_response_payload(
        payload,
        safe_mode=safe_mode,
        max_group_id=max_group_id,
        section_plans=section_plans,
    )
    response.object_plans = normalized.get("object_plans", [])
    response.trigger_plans = normalized.get("trigger_plans", [])
    response.sections = normalized.get("sections", [])
    response.gameplay_events = normalized.get("gameplay_events", [])
    response.speed_plan = normalized.get("speed_plan", [])
    response.metadata = dict(normalized.get("metadata", {})) if isinstance(normalized.get("metadata", {}), dict) else {}
    return report

def _extract_json_text(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()
    return text

def _list_of_dicts(value: Any, field_name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{field_name} must contain objects")
    return [dict(item) for item in value]

def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:  # noqa: BLE001
        return False

def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:  # noqa: BLE001
        return None
    return result if math.isfinite(result) else None

def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return _float_or_none(value)

def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return None

def _positive_intish(value: Any) -> bool:
    try:
        return int(value) > 0
    except Exception:  # noqa: BLE001
        return False