# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from gmdgen.gd.time_mapping import SpeedState, normalize_speed_state


@dataclass(slots=True)
class AudioEvent:
    time: float
    beat_index: int
    is_downbeat: bool
    onset_strength: float
    energy: float
    section_id: int
    section_type: str
    confidence: float = 1.0


@dataclass(slots=True)
class GDTimeEvent:
    time: float
    x: float
    speed_state: SpeedState
    event_type: str
    importance: float
    nearest_beat_time: float
    sync_error: float


@dataclass(slots=True)
class LevelSettingsPlan:
    game_mode: str = "cube"
    speed: SpeedState = SpeedState.NORMAL
    mini: bool = False
    dual: bool = False
    two_player: bool = False
    song_offset: float = 0.0
    fade_in: bool = False
    fade_out: bool = False
    background_id: int = 1
    ground_id: int = 1
    font_id: int = 1
    line_id: int = 1
    guideline_string: str = ""
    custom_song: bool = True


@dataclass(slots=True)
class SectionPlan:
    start_time: float
    end_time: float
    start_x: float
    end_x: float
    section_type: str
    gameplay_mode: str
    speed_state: SpeedState
    density_target: float
    decoration_intensity: float
    trigger_intensity: float
    difficulty_target: float

    @property
    def density(self) -> float:
        return self.density_target


@dataclass(slots=True)
class GameplayEvent:
    time: float
    x: float
    event_type: str
    importance: float
    beat_index: int
    sync_error: float
    section_id: int


@dataclass(slots=True)
class ObjectPlan:
    object_id: str
    x: float
    y: float
    role: str
    group_ids: list[int] = field(default_factory=list)
    color_channel: int | None = None
    trigger_target_group: int | None = None
    duration: float | None = None
    spawn_delay: float | None = None
    editor_layer: int | None = None
    z_layer: int | None = None
    z_order: int | None = None
    rotation: float = 0.0
    scale: float = 1.0
    safety_flags: dict[str, Any] = field(default_factory=dict)
    beat_aligned_time: float | None = None
    sync_error: float = 0.0


@dataclass(slots=True)
class TriggerPlan:
    trigger_type: str
    object_id: str
    x: float
    y: float
    target_group: int | None = None
    secondary_group: int | None = None
    duration: float = 0.0
    easing: str | None = None
    spawn_delay: float = 0.0
    multi_trigger: bool = False
    editor_disable: bool = False
    beat_aligned_time: float | None = None
    sync_error: float = 0.0
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PlayabilityWarning:
    warning_type: str
    severity: str
    time: float | None
    x: float | None
    mode: str
    speed_state: str
    message: str
    related_event_ids: list[int] = field(default_factory=list)


@dataclass(slots=True)
class PlanCountReport:
    raw_ai_objects: int | None = None
    raw_ai_triggers: int | None = None
    parsed_objects: int | None = None
    parsed_triggers: int | None = None
    normalized_objects: int | None = None
    normalized_triggers: int | None = None
    repaired_objects: int | None = None
    repaired_triggers: int | None = None
    rendered_objects: int | None = None
    rendered_triggers: int | None = None
    final_encoded_objects: int | None = None
    final_encoded_triggers: int | None = None
    selected_candidate_objects: int | None = None
    selected_candidate_triggers: int | None = None
    actual_saved_objects: int | None = None
    actual_saved_triggers: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        return {
            "raw_ai_objects": self.raw_ai_objects,
            "raw_ai_triggers": self.raw_ai_triggers,
            "parsed_objects": self.parsed_objects,
            "parsed_triggers": self.parsed_triggers,
            "normalized_objects": self.normalized_objects,
            "normalized_triggers": self.normalized_triggers,
            "repaired_objects": self.repaired_objects,
            "repaired_triggers": self.repaired_triggers,
            "rendered_objects": self.rendered_objects,
            "rendered_triggers": self.rendered_triggers,
            "final_encoded_objects": self.final_encoded_objects,
            "final_encoded_triggers": self.final_encoded_triggers,
            "selected_candidate_objects": self.selected_candidate_objects,
            "selected_candidate_triggers": self.selected_candidate_triggers,
            "actual_saved_objects": self.actual_saved_objects,
            "actual_saved_triggers": self.actual_saved_triggers,
        }


@dataclass(slots=True)
class ValidationReport:
    generation_mode: str = "audio_conditioned"
    beat_sync_avg_error: float = 0.0
    onset_sync_avg_error: float = 0.0
    time_x_avg_error: float = 0.0
    time_x_max_error: float = 0.0
    orphan_trigger_count: int = 0
    invalid_group_count: int = 0
    object_budget_exceeded: bool = False
    overcrowded_sections: list[int] = field(default_factory=list)
    playability_warnings: list[str] = field(default_factory=list)
    editor_validity_warnings: list[str] = field(default_factory=list)
    final_object_count: int = 0
    score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    audio_file: str = ""
    audio_file_name: str = ""
    audio_backend: str = ""
    detected_bpm: float = 0.0
    beat_count: int = 0
    onset_count: int = 0
    section_count: int = 0
    speed_object_count: int = 0
    generated_trigger_count: int = 0
    round_trip_valid: bool = False
    editor_safety_report: dict[str, Any] = field(default_factory=dict)
    valid: bool = True
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float | int | str] = field(default_factory=dict)
    ai_provider: str = "ollama"
    ai_model: str = ""
    ai_used: bool = False
    ai_fallback_used: bool = False
    ai_fallback_reason: str = ""
    ai_context_chunks_used: int = 0
    ai_response_valid: bool = False
    ai_output_object_count: int = 0
    ai_output_trigger_count: int = 0
    ai_debug_artifact_path: str = ""
    ai_required: bool = True
    ai_error_type: str = ""
    ai_error_message_sanitized: str = ""
    ai_retry_count: int = 0
    ollama_sanitized_params: list[str] = field(default_factory=list)
    ollama_removed_unsupported_params: list[str] = field(default_factory=list)
    ollama_param_retry_count: int = 0
    ollama_model_capabilities: dict[str, Any] = field(default_factory=dict)
    provider_chain: dict[str, Any] | list[Any] = field(default_factory=dict)
    ai_calls_used: int = 0
    ai_cache_hits: int = 0
    ai_request_hashes: list[str] = field(default_factory=list)
    local_fallback_used: bool = False
    ollama_only_audit_passed: bool = False
    ai_normalization_warnings: list[str] = field(default_factory=list)
    ai_planning_used: bool = True
    ai_planning_error: str = ""
    deterministic_fallback_used: bool = False
    pruned_trigger_property_count: int = 0
    ignored_irrelevant_trigger_property_count: int = 0
    materialized_trigger_intent_count: int = 0
    auto_assigned_target_group_count: int = 0
    unresolved_missing_target_group_count: int = 0
    normalized_object_role_count: int = 0
    normalized_easing_count: int = 0
    fatal_validation_issue_count: int = 0
    nonfatal_validation_warning_count: int = 0
    geode_available: bool = False
    geode_version: str = ""
    geode_time_x_checked: bool = False
    geode_time_x_avg_error: float = 0.0
    geode_time_x_max_error: float = 0.0
    geode_parity_passed: bool = False
    geode_warnings: list[str] = field(default_factory=list)
    plan_snapshots: list[dict[str, Any]] = field(default_factory=list)
    plan_diffs: list[dict[str, Any]] = field(default_factory=list)
    raw_ai_object_count: int = 0
    raw_ai_trigger_count: int = 0
    final_trigger_count: int = 0
    removed_object_ratio: float = 0.0
    removed_trigger_ratio: float = 0.0
    repair_quality_report: dict[str, Any] = field(default_factory=dict)
    quality_loss_reason_summary: list[str] = field(default_factory=list)
    repair_loss_breakdown: dict[str, Any] = field(default_factory=dict)
    playability_breakdown: dict[str, Any] = field(default_factory=dict)
    density_target_by_section: dict[str, float] = field(default_factory=dict)
    actual_density_by_section: dict[str, float] = field(default_factory=dict)
    density_target_error: float = 0.0
    drop_impact_score: float = 0.0
    buildup_progression_score: float = 0.0
    candidate_reports: list[dict[str, Any]] = field(default_factory=list)
    selected_candidate_id: int = 0
    section_candidate_reports: list[dict[str, Any]] = field(default_factory=list)
    selected_section_count: int = 0
    global_consistency_report: dict[str, Any] = field(default_factory=dict)
    quality_gate_report: dict[str, Any] = field(default_factory=dict)
    quality_mode: str = "Balanced"
    plan_count_report: PlanCountReport = field(default_factory=PlanCountReport)

    def add_issue(self, message: str) -> None:
        self.valid = False
        self.issues.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation_mode": self.generation_mode,
            "beat_sync_avg_error": self.beat_sync_avg_error,
            "onset_sync_avg_error": self.onset_sync_avg_error,
            "time_x_avg_error": self.time_x_avg_error,
            "time_x_max_error": self.time_x_max_error,
            "orphan_trigger_count": self.orphan_trigger_count,
            "invalid_group_count": self.invalid_group_count,
            "object_budget_exceeded": self.object_budget_exceeded,
            "overcrowded_sections": list(self.overcrowded_sections),
            "playability_warnings": list(self.playability_warnings),
            "editor_validity_warnings": list(self.editor_validity_warnings),
            "final_object_count": self.final_object_count,
            "score": self.score,
            "score_breakdown": dict(self.score_breakdown),
            "audio_file": self.audio_file,
            "audio_file_name": self.audio_file_name,
            "audio_backend": self.audio_backend,
            "detected_bpm": self.detected_bpm,
            "beat_count": self.beat_count,
            "onset_count": self.onset_count,
            "section_count": self.section_count,
            "speed_object_count": self.speed_object_count,
            "generated_trigger_count": self.generated_trigger_count,
            "round_trip_valid": self.round_trip_valid,
            "editor_safety_report": dict(self.editor_safety_report),
            "valid": self.valid,
            "issues": list(self.issues),
            "warnings": list(self.warnings),
            "metrics": dict(self.metrics),
            "ai_provider": self.ai_provider,
            "ai_model": self.ai_model,
            "ai_used": self.ai_used,
            "ai_fallback_used": self.ai_fallback_used,
            "ai_fallback_reason": self.ai_fallback_reason,
            "ai_context_chunks_used": self.ai_context_chunks_used,
            "ai_response_valid": self.ai_response_valid,
            "ai_output_object_count": self.ai_output_object_count,
            "ai_output_trigger_count": self.ai_output_trigger_count,
            "ai_debug_artifact_path": self.ai_debug_artifact_path,
            "ai_required": self.ai_required,
            "ai_error_type": self.ai_error_type,
            "ai_error_message_sanitized": self.ai_error_message_sanitized,
            "ai_retry_count": self.ai_retry_count,
            "ollama_sanitized_params": list(self.ollama_sanitized_params),
            "ollama_removed_unsupported_params": list(self.ollama_removed_unsupported_params),
            "ollama_param_retry_count": self.ollama_param_retry_count,
            "ollama_model_capabilities": dict(self.ollama_model_capabilities),
            "provider_chain": self.provider_chain,
            "ai_calls_used": self.ai_calls_used,
            "ai_cache_hits": self.ai_cache_hits,
            "ai_request_hashes": list(self.ai_request_hashes),
            "local_fallback_used": self.local_fallback_used,
            "ollama_only_audit_passed": self.ollama_only_audit_passed,
            "ai_normalization_warnings": list(self.ai_normalization_warnings),
            "pruned_trigger_property_count": self.pruned_trigger_property_count,
            "ignored_irrelevant_trigger_property_count": self.ignored_irrelevant_trigger_property_count,
            "materialized_trigger_intent_count": self.materialized_trigger_intent_count,
            "auto_assigned_target_group_count": self.auto_assigned_target_group_count,
            "unresolved_missing_target_group_count": self.unresolved_missing_target_group_count,
            "normalized_object_role_count": self.normalized_object_role_count,
            "normalized_easing_count": self.normalized_easing_count,
            "fatal_validation_issue_count": self.fatal_validation_issue_count,
            "nonfatal_validation_warning_count": self.nonfatal_validation_warning_count,
            "geode_available": self.geode_available,
            "geode_version": self.geode_version,
            "geode_time_x_checked": self.geode_time_x_checked,
            "geode_time_x_avg_error": self.geode_time_x_avg_error,
            "geode_time_x_max_error": self.geode_time_x_max_error,
            "geode_parity_passed": self.geode_parity_passed,
            "geode_warnings": list(self.geode_warnings),
            "plan_snapshots": list(self.plan_snapshots),
            "plan_diffs": list(self.plan_diffs),
            "raw_ai_object_count": self.raw_ai_object_count,
            "raw_ai_trigger_count": self.raw_ai_trigger_count,
            "final_trigger_count": self.final_trigger_count,
            "removed_object_ratio": self.removed_object_ratio,
            "removed_trigger_ratio": self.removed_trigger_ratio,
            "repair_quality_report": dict(self.repair_quality_report),
            "quality_loss_reason_summary": list(self.quality_loss_reason_summary),
            "repair_loss_breakdown": dict(self.repair_loss_breakdown),
            "playability_breakdown": dict(self.playability_breakdown),
            "density_target_by_section": dict(self.density_target_by_section),
            "actual_density_by_section": dict(self.actual_density_by_section),
            "density_target_error": self.density_target_error,
            "drop_impact_score": self.drop_impact_score,
            "buildup_progression_score": self.buildup_progression_score,
            "candidate_reports": list(self.candidate_reports),
            "selected_candidate_id": self.selected_candidate_id,
            "section_candidate_reports": list(self.section_candidate_reports),
            "selected_section_count": self.selected_section_count,
            "global_consistency_report": dict(self.global_consistency_report),
            "quality_gate_report": dict(self.quality_gate_report),
            "quality_mode": self.quality_mode,
            "plan_count_report": self.plan_count_report.to_dict(),
        }


TRIGGER_OBJECT_IDS: dict[str, str] = {
    "move": "901",
    "pulse": "1006",
    "alpha": "1007",
    "spawn": "1268",
    "stop": "899",
    "touch": "1595",
    "count": "1611",
    "collision": "1815",
    "pickup": "1817",
    "follow": "1347",
    "follow_player_y": "1814",
    "shake": "1520",
    "color": "29",
    "bg_color": "29",
    "ground_color": "29",
    "line_color": "29",
}


class TriggerMode(str, Enum):
    SAFE = "safe"
    ADVANCED = "advanced"


@dataclass(slots=True, frozen=True)
class TriggerSchema:
    trigger_type: str
    object_id: str
    required_fields: tuple[str, ...] = ()
    optional_fields: tuple[str, ...] = ()
    default_values: dict[str, Any] = field(default_factory=dict)
    valid_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    requires_target_group: bool = False
    requires_duration: bool = False
    supports_spawn_delay: bool = False
    safe_mode_allowed: bool = False
    advanced_mode_allowed: bool = True


TRIGGER_SCHEMAS: dict[str, TriggerSchema] = {
    "move": TriggerSchema(
        trigger_type="move",
        object_id="901",
        required_fields=("target_group",),
        optional_fields=("duration", "easing"),
        default_values={"duration": 0.25, "easing": "linear"},
        valid_ranges={"duration": (0.0, 8.0)},
        requires_target_group=True,
        requires_duration=True,
        safe_mode_allowed=True,
    ),
    "pulse": TriggerSchema(
        trigger_type="pulse",
        object_id="1006",
        required_fields=("target_group",),
        optional_fields=("duration",),
        default_values={"duration": 0.18},
        valid_ranges={"duration": (0.02, 2.0)},
        requires_target_group=True,
        requires_duration=True,
        safe_mode_allowed=True,
    ),
    "alpha": TriggerSchema(
        trigger_type="alpha",
        object_id="1007",
        required_fields=("target_group",),
        optional_fields=("duration",),
        default_values={"duration": 0.2},
        valid_ranges={"duration": (0.02, 2.0)},
        requires_target_group=True,
        requires_duration=True,
        safe_mode_allowed=True,
    ),
    "spawn": TriggerSchema(
        trigger_type="spawn",
        object_id="1268",
        required_fields=("target_group",),
        optional_fields=("spawn_delay",),
        default_values={"spawn_delay": 0.0},
        valid_ranges={"spawn_delay": (0.0, 8.0)},
        requires_target_group=True,
        supports_spawn_delay=True,
        safe_mode_allowed=True,
    ),
    "stop": TriggerSchema(
        trigger_type="stop",
        object_id="899",
        required_fields=("target_group",),
        requires_target_group=True,
        safe_mode_allowed=True,
    ),
    "follow": TriggerSchema(
        trigger_type="follow",
        object_id="1347",
        required_fields=("target_group", "secondary_group"),
        optional_fields=("duration",),
        default_values={"duration": 0.5},
        valid_ranges={"duration": (0.02, 8.0)},
        requires_target_group=True,
        requires_duration=True,
        safe_mode_allowed=False,
    ),
    "shake": TriggerSchema(
        trigger_type="shake",
        object_id="1520",
        optional_fields=("duration",),
        default_values={"duration": 0.12},
        valid_ranges={"duration": (0.02, 1.5)},
        requires_duration=True,
        safe_mode_allowed=False,
    ),
    "color": TriggerSchema(
        trigger_type="color",
        object_id="29",
        optional_fields=("duration", "target_group"),
        default_values={"duration": 0.0},
        valid_ranges={"duration": (0.0, 8.0)},
        safe_mode_allowed=True,
    ),
    "toggle": TriggerSchema(
        trigger_type="toggle",
        object_id="33",
        required_fields=("target_group",),
        requires_target_group=True,
        safe_mode_allowed=True,
    ),
    "count": TriggerSchema(
        trigger_type="count",
        object_id="1611",
        required_fields=("target_group",),
        requires_target_group=True,
        safe_mode_allowed=False,
    ),
    "collision": TriggerSchema(
        trigger_type="collision",
        object_id="1815",
        required_fields=("target_group",),
        requires_target_group=True,
        safe_mode_allowed=False,
    ),
    "pickup": TriggerSchema(
        trigger_type="pickup",
        object_id="1817",
        required_fields=("target_group",),
        requires_target_group=True,
        safe_mode_allowed=False,
    ),
}


def normalize_trigger_mode(mode: TriggerMode | str | bool | None) -> TriggerMode:
    if isinstance(mode, TriggerMode):
        return mode
    if mode is True:
        return TriggerMode.SAFE
    if mode is False or mode is None:
        return TriggerMode.ADVANCED
    value = str(mode).strip().lower()
    return TriggerMode.SAFE if value == "safe" else TriggerMode.ADVANCED


def get_trigger_schema(trigger_type: str) -> TriggerSchema | None:
    return TRIGGER_SCHEMAS.get(str(trigger_type).strip().lower())


def is_trigger_allowed_in_mode(trigger_type: str, mode: TriggerMode | str | bool | None) -> bool:
    schema = get_trigger_schema(trigger_type)
    if schema is None:
        return False
    normalized = normalize_trigger_mode(mode)
    if normalized == TriggerMode.SAFE:
        return schema.safe_mode_allowed
    return schema.advanced_mode_allowed


def _has_trigger_field(plan: TriggerPlan, field_name: str) -> bool:
    value = getattr(plan, field_name)
    if field_name in {"target_group", "secondary_group"}:
        return value is not None
    if field_name in {"duration", "spawn_delay"}:
        return value is not None and float(value) >= 0.0
    return value is not None


def apply_trigger_defaults(
    trigger_plan: TriggerPlan,
    mode: TriggerMode | str | bool | None = TriggerMode.SAFE,
) -> TriggerPlan | None:
    schema = get_trigger_schema(trigger_plan.trigger_type)
    if schema is None or not is_trigger_allowed_in_mode(trigger_plan.trigger_type, mode):
        return None
    trigger_plan.object_id = schema.object_id
    for field_name, value in schema.default_values.items():
        if not _has_trigger_field(trigger_plan, field_name):
            setattr(trigger_plan, field_name, value)
    for field_name, (low, high) in schema.valid_ranges.items():
        value = getattr(trigger_plan, field_name)
        if value is None:
            continue
        setattr(trigger_plan, field_name, max(low, min(high, float(value))))
    return trigger_plan


def validate_trigger_plan_schema(
    trigger_plan: TriggerPlan,
    mode: TriggerMode | str | bool | None = TriggerMode.SAFE,
    *,
    max_group_id: int | None = None,
) -> list[str]:
    schema = get_trigger_schema(trigger_plan.trigger_type)
    if schema is None:
        return [f"unsupported_trigger_schema: {trigger_plan.trigger_type}"]
    issues: list[str] = []
    if not is_trigger_allowed_in_mode(trigger_plan.trigger_type, mode):
        issues.append(f"trigger_not_allowed_in_mode: {trigger_plan.trigger_type}")
    if str(trigger_plan.object_id) != str(schema.object_id):
        issues.append(
            f"trigger_object_id_mismatch: {trigger_plan.trigger_type} uses {trigger_plan.object_id}, expected {schema.object_id}"
        )
    for field_name in schema.required_fields:
        if not _has_trigger_field(trigger_plan, field_name):
            issues.append(f"trigger_missing_required_field: {trigger_plan.trigger_type}.{field_name}")
    if schema.requires_target_group and trigger_plan.target_group is None:
        issues.append(f"trigger_missing_target_group: {trigger_plan.trigger_type}")
    if schema.requires_duration and trigger_plan.duration <= 0:
        issues.append(f"trigger_missing_duration: {trigger_plan.trigger_type}")
    for field_name, (low, high) in schema.valid_ranges.items():
        value = getattr(trigger_plan, field_name)
        if value is not None and not (low <= float(value) <= high):
            issues.append(f"trigger_range_violation: {trigger_plan.trigger_type}.{field_name}")
    if max_group_id is not None:
        for field_name in ("target_group", "secondary_group"):
            value = getattr(trigger_plan, field_name)
            if value is not None and not (1 <= int(value) <= max_group_id):
                issues.append(f"trigger_group_bounds: {trigger_plan.trigger_type}.{field_name}")
    return issues


def build_level_settings(
    *,
    start_speed: SpeedState | str | None = SpeedState.NORMAL,
    song_offset: float = 0.0,
    game_mode: str = "cube",
    custom_song: bool = True,
    guideline_string: str = "",
) -> LevelSettingsPlan:
    return LevelSettingsPlan(
        game_mode=game_mode,
        speed=normalize_speed_state(start_speed),
        song_offset=float(song_offset),
        custom_song=custom_song,
        guideline_string=guideline_string,
    )


def _format_num(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    rounded = round(float(value), 5)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.5f}".rstrip("0").rstrip(".")


def _append_pair(parts: list[str], key: str, value: str | int | float | bool | None) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        value = 1 if value else 0
    parts.extend([key, _format_num(value) if isinstance(value, (int, float)) else str(value)])


def object_plan_to_level_object(plan: ObjectPlan) -> str:
    """Conservative save-string encoder for planned normal objects.

    It only emits stable keys already used elsewhere in this codebase:
    1 object id, 2 x, 3 y, 6 rotation, 20 editor layer, 24 z layer,
    25 z order, 32 scale, 155 group ids.
    """

    parts = ["1", str(plan.object_id), "2", _format_num(plan.x), "3", _format_num(plan.y)]
    if abs(plan.rotation) > 1e-6:
        _append_pair(parts, "6", plan.rotation)
    if abs(plan.scale - 1.0) > 1e-6:
        _append_pair(parts, "32", plan.scale)
    _append_pair(parts, "20", plan.editor_layer)
    _append_pair(parts, "24", plan.z_layer)
    _append_pair(parts, "25", plan.z_order)
    if plan.group_ids:
        parts.extend(["155", ".".join(str(group_id) for group_id in sorted(set(plan.group_ids)))])
    return ",".join(parts)


def trigger_plan_to_level_object(plan: TriggerPlan) -> str:
    """Conservative encoder for EffectGameObject-style triggers.

    duration uses key 10 and target group uses key 51, both observed in the
    local decoded samples. More exotic fields remain in TriggerPlan until a
    version-specific key map is proven.
    """

    parts = ["1", str(plan.object_id), "2", _format_num(plan.x), "3", _format_num(plan.y)]
    if plan.target_group is not None:
        _append_pair(parts, "51", plan.target_group)
    if plan.secondary_group is not None:
        _append_pair(parts, "71", plan.secondary_group)
    if plan.duration > 0:
        _append_pair(parts, "10", plan.duration)
    schema = get_trigger_schema(plan.trigger_type)
    if schema and schema.supports_spawn_delay and plan.spawn_delay > 0:
        _append_pair(parts, "63", plan.spawn_delay)
    if plan.multi_trigger:
        _append_pair(parts, "35", 1)
    if plan.editor_disable:
        _append_pair(parts, "58", 1)
    return ",".join(parts)


def plans_to_level_objects(
    object_plans: list[ObjectPlan],
    trigger_plans: list[TriggerPlan],
    *,
    trigger_mode: TriggerMode | str | bool | None = TriggerMode.ADVANCED,
) -> list[str]:
    encoded = [object_plan_to_level_object(plan) for plan in object_plans]
    encoded.extend(
        trigger_plan_to_level_object(plan)
        for plan in trigger_plans
        if not validate_trigger_plan_schema(plan, trigger_mode)
    )
    return sorted(
        encoded,
        key=lambda raw: (
            float(_extract_field(raw, "2") or 0.0),
            float(_extract_field(raw, "3") or 0.0),
        ),
    )


def _extract_field(level_object: str, key: str) -> str | None:
    parts = level_object.split(",")
    for idx in range(0, len(parts) - 1, 2):
        if parts[idx] == key:
            return parts[idx + 1]
    return None
