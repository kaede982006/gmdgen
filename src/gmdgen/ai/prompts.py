from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from gmdgen.ai.normalization import allowed_object_roles
from gmdgen.ai.schemas import AI_LEVEL_PLAN_JSON_SCHEMA, AILevelPlanRequest
from gmdgen.gd.plans import SectionPlan
from gmdgen.gd.triggers import TRIGGER_SCHEMA_REGISTRY, list_supported_triggers


GLOBAL_PLANNER_PROMPT = """You are a Geometry Dash Global Planner.
Analyze the audio structure and output a high-level `sections` plan and `speed_plan`.
Define start_time, end_time, density_target, and trigger_intensity for each section.
Do not generate object_plans or trigger_plans.
Reflect intro, buildup, drop, break, and outro differently. Drop sections must have high density_target."""

SECTION_PLANNER_PROMPT = """You are a Geometry Dash Section Planner.
Analyze the specific section constraint and output a detailed `gameplay_events` rhythmic skeleton.
Align important gameplay events to strong beats. Do not generate raw objects yet."""

OBJECT_PLANNER_PROMPT = """You are a Geometry Dash Object Planner.
Given gameplay events and section constraints, output `object_plans` using allowed roles.
Do not output triggers. Use beat_orb and beat_pad for gameplay.
Keep density aligned with the section's density_target."""

TRIGGER_PLANNER_PROMPT = """You are a Geometry Dash Trigger Planner.
Given the object_plans, output `trigger_plans` to decorate and animate the level.
Respect safe mode constraints and schemas. Do not add structural objects."""

CRITIC_PROMPT = """You are a Geometry Dash Plan Critic.
Review the provided level plan against the audio and quality constraints.
Output `reasoning_summary` and `safety_notes` identifying any sync errors or empty drops."""

REVISION_PROMPT = """You are a Geometry Dash Plan Revisionist.
Fix the errors identified by the Critic and output the final corrected JSON plan.
Ensure empty drops are populated and out-of-bounds coordinates are fixed."""

SYSTEM_PROMPT = """You are a Geometry Dash level planning assistant.
Return only structured JSON matching the provided schema.
Do not output raw Geometry Dash save strings or arbitrary save keys.
Use the audio beat/onset/section/time-X summaries as the primary timing source.
Respect song offset, start speed, speed portals, trigger schema, group ids, object budget, and safe mode.
Every trigger that requires a target group must reference a group defined by an ObjectPlan.
Create a playable rhythmic layout, not a sparse object list.
Reflect intro, buildup, drop, break, and outro differently in gameplay and visuals.
Drop sections must not be empty; they should have higher valid density and trigger intensity.
Do not create empty drop sections.
Buildups should gradually increase density and tension. Breaks should provide space and easier patterns.
Repeat rhythmic motifs as repeated gameplay motifs with small variation.
Good plans align important gameplay events to strong beats and visual triggers to strong onsets.
Vary density according to RMS/energy and use section transitions near downbeats or strong boundaries.
Include readable structure objects, gameplay events, and style-appropriate decoration.
Avoid excessive repetition of one role or one object id.
Use learned style profiles, retrieved motifs, and previous feedback as inspiration when provided.
Prefer object distributions similar to learned high-quality examples while staying valid and playable.
For drop sections, prefer learned motifs tagged high_density or drop-like when available.
Avoid learned failure patterns from low-rated generations.
Do not copy raw level strings from learned data; only use summarized motifs and style hints.
Do not invent unsupported trigger properties or unknown GD 2.2 save keys.
Prefer trigger intent over low-level trigger properties: use properties.trigger_kind, purpose, target_role,
intensity, duration_hint, and section_id when you know the intended visual/gameplay effect.
The renderer/materializer will assign safe low-level trigger fields and target groups.
Use only the allowed ObjectPlan roles. Use beat_orb for orbs and beat_pad for jump pads.
For each trigger type, set irrelevant trigger properties to null instead of inventing cross-trigger fields.
Do not generate move_x or move_y for pulse, color, or alpha triggers.
Do not generate color_channel or opacity for move triggers.
Allowed easing values are linear, ease_in, ease_out, and ease_in_out.
In safe mode, use only safe trigger types and conservative durations.
Prefer fewer valid triggers over many invalid triggers.
Do not create long sections with no gameplay events.
Keep the reasoning_summary short; do not include private chain-of-thought."""


def summarize_audio_for_model(features: Any, *, max_beats: int = 16, max_onsets: int = 16) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    beat_items = getattr(features, "beats", None) or getattr(features, "beat_features", [])
    onset_items = getattr(features, "onsets", [])
    beat_summary = {
        "count": len(getattr(features, "beat_times", [])),
        "representative": [
            {
                "time": round(float(getattr(beat, "time", 0.0)), 4),
                "index": int(getattr(beat, "index", idx)),
                "strength": round(float(getattr(beat, "strength", 0.0)), 4),
                "is_downbeat": bool(getattr(beat, "is_downbeat", False)),
            }
            for idx, beat in enumerate(_sample_evenly(beat_items, max_beats))
        ],
    }
    strongest_onsets = sorted(
        onset_items,
        key=lambda onset: float(getattr(onset, "strength", 0.0)),
        reverse=True,
    )[:max_onsets]
    onset_summary = {
        "count": len(getattr(features, "onset_times", [])),
        "top_peaks": [
            {
                "time": round(float(getattr(onset, "time", 0.0)), 4),
                "strength": round(float(getattr(onset, "strength", 0.0)), 4),
            }
            for onset in strongest_onsets
        ],
    }
    audio_summary = {
        "duration": round(float(getattr(features, "duration", 0.0)), 4),
        "sample_rate": int(getattr(features, "sample_rate", 0) or 0),
        "bpm": round(float(getattr(features, "bpm", 0.0)), 4),
        "backend": str(getattr(features, "backend", "")),
        "confidence": round(float(getattr(features, "confidence", 0.0)), 4),
    }
    return audio_summary, beat_summary, onset_summary


def summarize_sections_for_model(section_plans: list[SectionPlan], *, max_sections: int = 16) -> list[dict[str, Any]]:
    result = []
    for idx, section in enumerate(section_plans[:max_sections]):
        result.append(
            {
                "section_id": idx,
                "start_time": round(section.start_time, 4),
                "end_time": round(section.end_time, 4),
                "start_x": round(section.start_x, 2),
                "end_x": round(section.end_x, 2),
                "section_type": section.section_type,
                "gameplay_mode": section.gameplay_mode,
                "speed_state": section.speed_state.value,
                "density_target": round(section.density_target, 4),
                "trigger_intensity": round(section.trigger_intensity, 4),
                "difficulty_target": round(section.difficulty_target, 4),
            }
        )
    return result


def summarize_trigger_schema_for_model(*, safe_mode: bool) -> dict[str, Any]:
    mode = "safe" if safe_mode else "advanced"
    allowed = list_supported_triggers(mode)
    schemas: dict[str, Any] = {}
    for trigger_type in allowed:
        schema = TRIGGER_SCHEMA_REGISTRY.get(trigger_type)
        if not schema:
            continue
        schemas[trigger_type] = {
            "object_id": schema.object_id,
            "required_properties": list(schema.required_properties),
            "allowed_properties": [prop.name for prop in schema.properties],
            "preferred_intent_properties": [
                "trigger_kind",
                "purpose",
                "target_role",
                "intensity",
                "duration_hint",
                "section_id",
            ],
            "target_group_properties": list(schema.target_group_properties),
            "duration_properties": list(schema.duration_properties),
            "spawn_delay_properties": list(schema.spawn_delay_properties),
            "easing_properties": list(schema.easing_properties),
            "safe_mode_allowed": schema.safe_mode_allowed,
            "instruction": (
                "Prefer intent properties. Do not set properties outside allowed_properties; "
                "do not set low-level properties outside allowed_properties; "
                "use null for irrelevant generic fields. The materializer assigns target_group when target_role/section_id is clear."
            ),
        }
    return {"mode": mode, "allowed_triggers": allowed, "schemas": schemas}


def summarize_reference_style_for_model(style_profile: dict[str, Any], *, max_ids: int = 20) -> dict[str, Any]:
    ids_by_class = style_profile.get("ids_by_class", {}) if isinstance(style_profile, dict) else {}
    object_id_distribution = style_profile.get("object_id_distribution", {}) if isinstance(style_profile, dict) else {}
    role_distribution = style_profile.get("role_distribution", {}) if isinstance(style_profile, dict) else {}
    return {
        "object_classes": {
            str(class_name): [str(item) for item in values[:max_ids]]
            for class_name, values in ids_by_class.items()
            if isinstance(values, list)
        },
        "object_count": style_profile.get("object_count", 0) if isinstance(style_profile, dict) else 0,
        "trigger_count": style_profile.get("trigger_count", 0) if isinstance(style_profile, dict) else 0,
        "structure_object_ratio": style_profile.get("structure_object_ratio", 0.0) if isinstance(style_profile, dict) else 0.0,
        "decoration_object_ratio": style_profile.get("decoration_object_ratio", 0.0) if isinstance(style_profile, dict) else 0.0,
        "gameplay_object_ratio": style_profile.get("gameplay_object_ratio", 0.0) if isinstance(style_profile, dict) else 0.0,
        "trigger_ratio": style_profile.get("trigger_ratio", 0.0) if isinstance(style_profile, dict) else 0.0,
        "object_id_distribution": dict(list(object_id_distribution.items())[:20]) if isinstance(object_id_distribution, dict) else {},
        "role_distribution": dict(list(role_distribution.items())[:20]) if isinstance(role_distribution, dict) else {},
        "common_motif_patterns": style_profile.get("common_motif_patterns", [])[:8] if isinstance(style_profile, dict) else [],
        "has_level_header": bool(style_profile.get("level_header")) if isinstance(style_profile, dict) else False,
    }


def summarize_json_schema_for_model() -> dict[str, Any]:
    schema = AI_LEVEL_PLAN_JSON_SCHEMA["schema"]
    return {
        "required": schema["required"],
        "top_level_keys": sorted(schema["properties"].keys()),
        "allowed_object_roles": list(allowed_object_roles()),
        "raw_save_string_allowed": False,
    }


def truncate_context_to_budget(chunks: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    remaining = max(0, int(max_chars))
    result: list[dict[str, Any]] = []
    for chunk in chunks:
        text = str(chunk.get("text", ""))
        if remaining <= 0:
            break
        clipped = text[:remaining]
        copied = dict(chunk)
        copied["text"] = clipped
        result.append(copied)
        remaining -= len(clipped)
    return result


def build_compact_context_for_provider(
    request: AILevelPlanRequest,
    *,
    max_prompt_chars: int = 6000,
    top_k_context: int = 6,
) -> AILevelPlanRequest:
    """Return a compact copy for low-cost providers.

    The compact request keeps audio/section/schema essentials and trims bulky
    retrieved context/motifs without dumping raw dataset or .gmd content.
    """
    payload = request.to_dict()
    payload["retrieved_context"] = truncate_context_to_budget(
        list(request.retrieved_context)[:top_k_context],
        max(0, int(max_prompt_chars) // 3),
    )
    payload["retrieved_motifs"] = list(request.retrieved_motifs)[: min(8, top_k_context)]
    payload["learned_failure_patterns"] = list(request.learned_failure_patterns)[:6]
    payload["learned_success_patterns"] = list(request.learned_success_patterns)[:6]
    return AILevelPlanRequest(**payload)


def quality_criteria_for_model() -> list[str]:
    return [
        "align important gameplay events to strong beats",
        "align visual triggers to strong onsets",
        "vary density according to RMS/energy",
        "use section transitions at downbeats or strong section boundaries",
        "preserve playability safety margins",
        "include enough structure objects to be visually readable",
        "include enough gameplay events to avoid empty sections",
        "include decoration and visual accents proportional to style settings",
        "avoid overusing the same object role repeatedly",
        "use higher trigger intensity in drop sections",
        "keep trigger properties within allowed schema",
        "prefer fewer valid triggers over many invalid triggers",
        "do not create empty drop sections",
        "do not create long sections with no gameplay events",
    ]


def few_shot_plan_examples() -> list[dict[str, Any]]:
    return [
        {
            "case": "low_energy_intro",
            "sections": [{"section_id": 0, "section_type": "intro", "density_target": 0.25, "trigger_intensity": 0.1}],
            "object_plans": [
                {"object_id": 1, "x": 120, "y": 90, "role": "ai_structure"},
                {"object_id": 500, "x": 240, "y": 240, "role": "safe_decoration"},
            ],
            "trigger_plans": [],
        },
        {
            "case": "buildup",
            "sections": [{"section_id": 1, "section_type": "buildup", "density_target": 0.55, "trigger_intensity": 0.45}],
            "object_plans": [
                {"object_id": 1, "x": 420, "y": 90, "role": "ai_structure"},
                {"object_id": 36, "x": 510, "y": 180, "role": "beat_orb"},
            ],
            "trigger_plans": [
                {"trigger_type": "alpha", "object_id": "1007", "x": 420, "y": 300, "target_group": 1, "duration": 0.18, "properties": None}
            ],
        },
        {
            "case": "drop",
            "sections": [{"section_id": 2, "section_type": "drop", "density_target": 0.9, "trigger_intensity": 0.85}],
            "object_plans": [
                {"object_id": 35, "x": 720, "y": 132, "role": "beat_pad"},
                {"object_id": 36, "x": 780, "y": 190, "role": "beat_orb"},
                {"object_id": 500, "x": 720, "y": 240, "role": "visual_accent_target", "group_ids": [1]},
            ],
            "trigger_plans": [
                {"trigger_type": "pulse", "object_id": "1006", "x": 720, "y": 300, "target_group": 1, "duration": 0.15, "properties": {"color_channel": 1}}
            ],
        },
        {
            "case": "break",
            "sections": [{"section_id": 3, "section_type": "break", "density_target": 0.2, "trigger_intensity": 0.15}],
            "object_plans": [{"object_id": 1, "x": 1020, "y": 90, "role": "ai_structure"}],
            "trigger_plans": [],
        },
    ]


def _sample_evenly(items: list[Any], max_items: int) -> list[Any]:
    if len(items) <= max_items:
        return list(items)
    if max_items <= 1:
        return [items[0]]
    step = (len(items) - 1) / (max_items - 1)
    return [items[round(idx * step)] for idx in range(max_items)]
