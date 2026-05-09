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

    @property
    def valid(self) -> bool:
        return self.plan is not None and not self.errors and not self.fallback_used

    def to_report_fields(self) -> dict[str, Any]:
        return {
            "planner_status": "fallback" if self.fallback_used else ("valid" if self.plan else "invalid"),
            "planner_fallback_used": self.fallback_used,
            "planner_fallback_reason": self.fallback_reason,
            "planner_errors": list(self.errors),
            "planner_raw_payload_preview": str(self.raw_payload)[:1000] if self.raw_payload else None,
            "planner_json_payload": self.json_payload if self.plan else None,
        }


def parse_ollama_section_plan(payload: str | dict[str, Any]) -> PlannerParseResult:
    """Parse the strict Ollama planner JSON contract.

    Ollama is only allowed to describe a level and sections with symbolic
    references. Concrete Geometry Dash object strings, group ids, color channel
    ids, scores, and validation verdicts are rejected here.
    """
    raw_payload = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    
    if isinstance(payload, str):
        if _looks_like_raw_gmd(payload):
            return PlannerParseResult(errors=["raw_gmd_output_rejected"], raw_payload=raw_payload)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            return PlannerParseResult(errors=[f"json_parse_failed:{exc.msg}"], raw_payload=raw_payload)
    elif isinstance(payload, dict):
        data = payload
    else:
        return PlannerParseResult(errors=["planner_output_must_be_json_object"], raw_payload=raw_payload)

    errors = _reject_forbidden_shape(data)
    if errors:
        return PlannerParseResult(errors=errors, raw_payload=raw_payload, json_payload=data)

    if not isinstance(data, dict):
        return PlannerParseResult(errors=["planner_output_must_be_json_object"], raw_payload=raw_payload)

    # Handle top-level aliases first so normalization can find them
    if "level_plan" not in data:
        if "plan" in data and isinstance(data["plan"], dict):
            data["level_plan"] = data.pop("plan")
        elif "level" in data and isinstance(data["level"], dict):
            data["level_plan"] = data.pop("level")

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
    if not isinstance(sections_payload, list):
        errors.append("sections_required")
    
    if errors:
        return PlannerParseResult(errors=errors, raw_payload=raw_payload, json_payload=data)

    level_errors = _validate_level_plan_payload(level_payload)
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
    
    if errors:
        return PlannerParseResult(errors=errors, raw_payload=raw_payload, json_payload=data)

    return PlannerParseResult(
        plan=LevelPlan(
            level_name=str(level_payload["level_name"]),
            difficulty=str(level_payload["difficulty"]).lower(),
            target_duration=float(level_payload["target_duration"]),
            object_budget=int(level_payload["object_budget"]),
            style=str(level_payload["style"]),
            sync_intensity=str(level_payload["sync_intensity"]).lower(),
            sections=sections,
        ),
        raw_payload=raw_payload,
        json_payload=data,
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
    fallback = build_template_level_plan(
        prompt=prompt,
        level_name=fallback_level_name,
        object_budget=object_budget,
    )
    res = PlannerParseResult(
        plan=fallback,
        errors=list(parsed.errors),
        fallback_used=True,
        fallback_reason="invalid_ollama_planner_output",
        raw_payload=parsed.raw_payload,
        json_payload=parsed.json_payload,
    )
    return res


def build_template_level_plan(
    *,
    prompt: str = "",
    level_name: str = "fallback_plan",
    object_budget: int = 500,
) -> LevelPlan:
    notes = prompt.strip()[:180] or "deterministic fallback section plan"
    section = SectionPlan(
        section_id="s001",
        time_start=0.0,
        time_end=8.0,
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
    )
    return LevelPlan(
        level_name=level_name,
        difficulty="normal",
        target_duration=30.0,
        object_budget=max(1, int(object_budget)),
        style="modern_glow",
        sync_intensity="medium",
        sections=[section],
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
    try:
        target_duration = float(payload.get("target_duration"))
        if target_duration <= 0 or target_duration > 600:
            errors.append("level_plan_target_duration_out_of_range")
    except (TypeError, ValueError):
        errors.append("level_plan_target_duration_invalid")
    try:
        object_budget = int(payload.get("object_budget"))
        if object_budget <= 0 or object_budget > 40000:
            errors.append("level_plan_object_budget_out_of_range")
    except (TypeError, ValueError):
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
    try:
        time_start = float(payload.get("time_start"))
        time_end = float(payload.get("time_end"))
        if time_start < 0 or time_end <= time_start:
            errors.append(f"{prefix}:time_range_invalid")
    except (TypeError, ValueError):
        time_start = 0.0
        time_end = 0.0
        errors.append(f"{prefix}:time_range_invalid")
    try:
        density = float(payload.get("density"))
        if density < 0.0 or density > 1.0:
            errors.append(f"{prefix}:density_out_of_range")
    except (TypeError, ValueError):
        density = 0.0
        errors.append(f"{prefix}:density_invalid")
    try:
        trigger_budget = int(payload.get("trigger_budget"))
        if trigger_budget < 0 or trigger_budget > 128:
            errors.append(f"{prefix}:trigger_budget_out_of_range")
    except (TypeError, ValueError):
        trigger_budget = 0
        errors.append(f"{prefix}:trigger_budget_invalid")

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
            allowed_object_families=[str(item) for item in allowed if item],
            forbidden_features=[str(item) for item in forbidden if item],
            trigger_budget=trigger_budget,
            group_symbols=[GroupSymbol(str(item)) for item in groups if item],
            color_symbols=[ColorSymbol(str(item)) for item in colors or [] if item],
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
        if "difficulty" in lp:
            diff = str(lp["difficulty"]).lower()
            diff = diff.replace(" gameplay", "").replace(" mode", "").replace(" difficulty", "")
            if "-" in diff:
                parts = diff.split("-")
                for p in parts:
                    if p in DIFFICULTIES:
                        diff = p
                        break
            lp["difficulty"] = diff

        if "sync_intensity" in lp:
            sync = str(lp["sync_intensity"]).lower()
            if sync == "moderate":
                lp["sync_intensity"] = "medium"
            elif sync == "intense":
                lp["sync_intensity"] = "high"
            elif sync == "calm":
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
                section["game_mode"] = str(section["game_mode"]).lower().replace(" gameplay", "").replace(" mode", "")
            
            if "speed" in section:
                speed_str = str(section["speed"]).lower()
                speed_str = speed_str.replace(" speed", "").replace(" speed", "").replace("x", "").strip()
                if speed_str in ["half", "0.5", "slow"]:
                    section["speed"] = "0.5x"
                elif speed_str in ["normal", "1", "1.0"]:
                    section["speed"] = "1x"
                elif speed_str in ["fast", "2", "2.0"]:
                    section["speed"] = "2x"
                elif speed_str in ["very fast", "3", "3.0"]:
                    section["speed"] = "3x"
                elif speed_str in ["super fast", "4", "4.0"]:
                    section["speed"] = "4x"
                else:
                    section["speed"] = f"{speed_str}x"
