# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from gmdgen.data.preprocess import extract_level_header, split_level_objects
from gmdgen.features.tokenizer import (
    extract_object_id,
    extract_object_field,
    extract_object_number,
)
from gmdgen.io.gmd_decoder import decode_level_data
from gmdgen.io.gmd_decoder import encode_level_data
from gmdgen.io.gmd_parser import parse_gmd_text
from gmdgen.gd.plans import (
    ObjectPlan,
    PlayabilityWarning,
    SectionPlan,
    TriggerPlan,
    get_trigger_schema,
    is_trigger_allowed_in_mode,
    validate_trigger_plan_schema,
)
from gmdgen.gd.time_mapping import SpeedState, normalize_speed_state

_GROUP_KEY = "155"
_TRIGGER_TARGET_KEY = "51"
_TRIGGER_IDS: frozenset[str] = frozenset(
    {"29", "30", "32", "33", "142", "105", "899",
     "901", "1006", "1007", "1049", "1268", "1346", "1347", "1520",
     "1595", "1611", "1616", "1815", "1817", "2067"}
)
_SPEED_PORTAL_IDS: frozenset[str] = frozenset({"200", "201", "202", "203", "1334"})
_GAMEMODE_PORTAL_IDS: frozenset[str] = frozenset(
    {"12", "13", "47", "111", "660", "745", "1331"}
)
_SAFE_TRIGGER_IDS: frozenset[str] = frozenset({"29", "32", "33", "899", "901", "1006", "1007", "1268"})
_TRIGGER_TYPE_BY_ID: dict[str, str] = {
    schema.object_id: trigger_type
    for trigger_type, schema in {
        name: get_trigger_schema(name)
        for name in (
            "move", "pulse", "alpha", "spawn", "stop", "follow", "shake",
            "color", "toggle", "count", "collision", "pickup",
        )
    }.items()
    if schema is not None
}

PLAYABILITY_SPACING_TABLE: dict[str, dict[SpeedState, float]] = {
    "cube": {
        SpeedState.SLOW: 34.0,
        SpeedState.NORMAL: 42.0,
        SpeedState.FAST: 50.0,
        SpeedState.FASTER: 57.0,
        SpeedState.FASTEST: 66.0,
    },
    "ship": {
        SpeedState.SLOW: 48.0,
        SpeedState.NORMAL: 64.0,
        SpeedState.FAST: 76.0,
        SpeedState.FASTER: 86.0,
        SpeedState.FASTEST: 98.0,
    },
    "ball": {
        SpeedState.SLOW: 42.0,
        SpeedState.NORMAL: 54.0,
        SpeedState.FAST: 64.0,
        SpeedState.FASTER: 73.0,
        SpeedState.FASTEST: 84.0,
    },
    "ufo": {
        SpeedState.SLOW: 44.0,
        SpeedState.NORMAL: 58.0,
        SpeedState.FAST: 69.0,
        SpeedState.FASTER: 78.0,
        SpeedState.FASTEST: 90.0,
    },
    "wave": {
        SpeedState.SLOW: 32.0,
        SpeedState.NORMAL: 40.0,
        SpeedState.FAST: 48.0,
        SpeedState.FASTER: 56.0,
        SpeedState.FASTEST: 64.0,
    },
    "robot": {
        SpeedState.SLOW: 44.0,
        SpeedState.NORMAL: 58.0,
        SpeedState.FAST: 69.0,
        SpeedState.FASTER: 79.0,
        SpeedState.FASTEST: 91.0,
    },
    "spider": {
        SpeedState.SLOW: 39.0,
        SpeedState.NORMAL: 52.0,
        SpeedState.FAST: 62.0,
        SpeedState.FASTER: 70.0,
        SpeedState.FASTEST: 81.0,
    },
}


def _collect_group_ids(objects: list[str]) -> set[int]:
    groups: set[int] = set()
    for obj in objects:
        raw = extract_object_field(obj, _GROUP_KEY)
        if not raw:
            continue
        for part in raw.split("."):
            part = part.strip()
            if part.isdigit():
                groups.add(int(part))
    return groups


def _check_x_monotone(objects: list[str]) -> list[str]:
    issues: list[str] = []
    prev_x: float | None = None
    violations = 0
    for obj in objects:
        x_val = extract_object_number(obj, "2")
        if x_val is None:
            continue
        if prev_x is not None and x_val < prev_x - 5:
            violations += 1
        prev_x = x_val
    if violations > 0:
        issues.append(f"x_monotone_violation: {violations} objects have decreasing x")
    return issues


def _check_orphan_triggers(objects: list[str]) -> list[str]:
    defined_groups = _collect_group_ids(objects)
    orphan_count = 0
    for obj in objects:
        obj_id = extract_object_id(obj)
        if obj_id not in _TRIGGER_IDS:
            continue
        raw_target = extract_object_field(obj, _TRIGGER_TARGET_KEY)
        if raw_target is None:
            continue
        raw_target = raw_target.strip()
        if raw_target.isdigit() and int(raw_target) not in defined_groups:
            orphan_count += 1
    issues: list[str] = []
    if orphan_count > 0:
        issues.append(
            f"orphan_trigger: {orphan_count} triggers reference non-existent groups"
        )
    return issues


def _check_speed_portals_sorted(objects: list[str]) -> list[str]:
    xs: list[float] = []
    for obj in objects:
        obj_id = extract_object_id(obj)
        if obj_id not in _SPEED_PORTAL_IDS:
            continue
        x_val = extract_object_number(obj, "2")
        if x_val is not None:
            xs.append(x_val)
    if any(next_x < prev_x for prev_x, next_x in zip(xs, xs[1:])):
        return ["speed_portal_order: speed portals must be sorted by x"]
    return []


def _check_trigger_durations(objects: list[str]) -> list[str]:
    invalid = 0
    suspicious = 0
    for obj in objects:
        obj_id = extract_object_id(obj)
        if obj_id not in _TRIGGER_IDS:
            continue
        duration = extract_object_number(obj, "10")
        if duration is None:
            continue
        if duration < 0:
            invalid += 1
        elif duration > 8.0:
            suspicious += 1
    issues: list[str] = []
    if invalid:
        issues.append(f"trigger_duration_invalid: {invalid} triggers have negative duration")
    if suspicious:
        issues.append(f"trigger_duration_suspicious: {suspicious} triggers exceed 8 seconds")
    return issues


def _check_spawn_delays(objects: list[str]) -> list[str]:
    invalid = 0
    suspicious = 0
    for obj in objects:
        obj_id = extract_object_id(obj)
        if obj_id not in _TRIGGER_IDS:
            continue
        delay = extract_object_number(obj, "63")
        if delay is None:
            continue
        if delay < 0:
            invalid += 1
        elif delay > 8.0:
            suspicious += 1
    issues: list[str] = []
    if invalid:
        issues.append(f"spawn_delay_invalid: {invalid} triggers have negative delay")
    if suspicious:
        issues.append(f"spawn_delay_suspicious: {suspicious} triggers exceed 8 seconds")
    return issues


def _check_group_bounds(objects: list[str], *, max_group_id: int | None) -> list[str]:
    if max_group_id is None:
        return []
    invalid = 0
    for group_id in _collect_group_ids(objects):
        if group_id <= 0 or group_id > max_group_id:
            invalid += 1
    return [f"group_id_bounds: {invalid} group ids outside 1..{max_group_id}"] if invalid else []


def _check_object_budget(objects: list[str], *, object_budget: int | None) -> list[str]:
    if object_budget is None or object_budget < 1 or len(objects) <= object_budget:
        return []
    return [f"object_budget_exceeded: {len(objects)} objects > budget {object_budget}"]


def _check_safe_mode_triggers(objects: list[str], *, safe_mode: bool) -> list[str]:
    if not safe_mode:
        return []
    unsafe = 0
    for obj in objects:
        obj_id = extract_object_id(obj)
        if obj_id in _TRIGGER_IDS and obj_id not in _SAFE_TRIGGER_IDS:
            unsafe += 1
    return [f"unsafe_trigger: {unsafe} triggers not allowed in safe_mode"] if unsafe else []


def _check_trigger_schema(objects: list[str], *, safe_mode: bool, max_group_id: int | None) -> list[str]:
    issues: list[str] = []
    mode = "safe" if safe_mode else "advanced"
    unknown = 0
    invalid = 0
    for obj in objects:
        obj_id = extract_object_id(obj)
        if obj_id not in _TRIGGER_IDS or obj_id in _SPEED_PORTAL_IDS:
            continue
        trigger_type = _TRIGGER_TYPE_BY_ID.get(str(obj_id))
        if trigger_type is None:
            unknown += 1
            continue
        plan = TriggerPlan(
            trigger_type=trigger_type,
            object_id=str(obj_id),
            x=extract_object_number(obj, "2") or 0.0,
            y=extract_object_number(obj, "3") or 0.0,
            target_group=_int_field(obj, "51"),
            secondary_group=_int_field(obj, "71"),
            duration=extract_object_number(obj, "10") or 0.0,
            spawn_delay=extract_object_number(obj, "63") or 0.0,
            multi_trigger=extract_object_field(obj, "35") == "1",
            editor_disable=extract_object_field(obj, "58") == "1",
        )
        schema_issues = validate_trigger_plan_schema(plan, mode, max_group_id=max_group_id)
        invalid += len(schema_issues)
    if unknown:
        issues.append(f"unsupported_trigger_schema: {unknown} triggers have no schema")
    if invalid:
        issues.append(f"trigger_schema_invalid: {invalid} schema violations")
    return issues


def _check_tight_spacing(objects: list[str], *, min_gap: float = 8.0) -> list[str]:
    xs = sorted(
        x for x in (extract_object_number(obj, "2") for obj in objects)
        if x is not None
    )
    tight = sum(1 for prev, current in zip(xs, xs[1:]) if current - prev < min_gap)
    return [f"tight_spacing: {tight} adjacent object gaps below {min_gap}px"] if tight else []


def _int_field(level_object: str, key: str) -> int | None:
    raw = extract_object_field(level_object, key)
    if raw is None:
        return None
    raw = raw.strip()
    return int(raw) if raw.isdigit() else None


def _difficulty_relief(difficulty: str | float | int) -> float:
    if isinstance(difficulty, (int, float)):
        value = max(0.0, min(1.0, float(difficulty)))
    else:
        value = {
            "easy": 0.1,
            "normal": 0.25,
            "hard": 0.45,
            "harder": 0.6,
            "insane": 0.75,
            "demon": 0.95,
        }.get(str(difficulty).lower(), 0.35)
    return 1.0 - min(0.45, value * 0.4)


def _section_for_x(section_plans: list[SectionPlan], x_value: float) -> SectionPlan | None:
    for section in section_plans:
        if section.start_x <= x_value < section.end_x:
            return section
    return section_plans[-1] if section_plans else None


def _plan_x(plan: ObjectPlan | dict) -> float | None:
    if isinstance(plan, dict):
        value = plan.get("x")
    else:
        value = getattr(plan, "x", None)
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _plan_role(plan: ObjectPlan | dict) -> str:
    return str(plan.get("role") if isinstance(plan, dict) else getattr(plan, "role", ""))


def _plan_object_id(plan: ObjectPlan | dict) -> str:
    return str(plan.get("object_id") if isinstance(plan, dict) else getattr(plan, "object_id", ""))


def validate_min_event_spacing(
    object_plans: list[ObjectPlan | dict],
    section_plans: list[SectionPlan],
    *,
    difficulty: str | float | int,
) -> list[PlayabilityWarning]:
    warnings: list[PlayabilityWarning] = []
    gameplay = [
        plan for plan in object_plans
        if "structure" in _plan_role(plan)
        or _plan_role(plan) in {"beat_orb", "beat_pad", "hazard", "ground_or_structure"}
    ]
    gameplay.sort(key=lambda plan: _plan_x(plan) or 0.0)
    relief = _difficulty_relief(difficulty)
    for index, (prev, current) in enumerate(zip(gameplay, gameplay[1:])):
        prev_x = _plan_x(prev)
        current_x = _plan_x(current)
        if prev_x is None or current_x is None:
            continue
        section = _section_for_x(section_plans, current_x)
        mode = section.gameplay_mode if section else "cube"
        speed = section.speed_state if section else SpeedState.NORMAL
        speed = normalize_speed_state(speed)
        min_gap = PLAYABILITY_SPACING_TABLE.get(mode, PLAYABILITY_SPACING_TABLE["cube"])[speed] * relief
        if current_x - prev_x < min_gap:
            warnings.append(
                PlayabilityWarning(
                    warning_type="min_event_spacing",
                    severity="warning",
                    time=None,
                    x=current_x,
                    mode=mode,
                    speed_state=speed.value,
                    message=f"{mode}/{speed.value} spacing {current_x - prev_x:.2f}px < {min_gap:.2f}px",
                    related_event_ids=[index, index + 1],
                )
            )
    return warnings


def validate_portal_safety_margin(
    object_plans: list[ObjectPlan | dict],
    section_plans: list[SectionPlan],
    *,
    difficulty: str | float | int,
) -> list[PlayabilityWarning]:
    warnings: list[PlayabilityWarning] = []
    relief = _difficulty_relief(difficulty)
    portal_ids = _SPEED_PORTAL_IDS | _GAMEMODE_PORTAL_IDS
    portals = [
        plan for plan in object_plans
        if _plan_object_id(plan) in portal_ids or _plan_role(plan) == "speed_portal"
    ]
    gameplay = [
        plan for plan in object_plans
        if "structure" in _plan_role(plan) or _plan_role(plan) in {"beat_orb", "beat_pad", "hazard"}
    ]
    for portal in portals:
        portal_x = _plan_x(portal)
        if portal_x is None:
            continue
        section = _section_for_x(section_plans, portal_x)
        mode = section.gameplay_mode if section else "cube"
        speed = normalize_speed_state(section.speed_state if section else SpeedState.NORMAL)
        margin = PLAYABILITY_SPACING_TABLE.get(mode, PLAYABILITY_SPACING_TABLE["cube"])[speed] * 1.35 * relief
        if any((x := _plan_x(plan)) is not None and portal_x < x < portal_x + margin for plan in gameplay):
            warnings.append(
                PlayabilityWarning(
                    warning_type="portal_safety_margin",
                    severity="warning",
                    time=None,
                    x=portal_x,
                    mode=mode,
                    speed_state=speed.value,
                    message=f"gameplay object too close after portal within {margin:.2f}px",
                )
            )
    return warnings


def validate_excessive_input_density(
    object_plans: list[ObjectPlan | dict],
    *,
    difficulty: str | float | int,
) -> list[PlayabilityWarning]:
    input_events = [
        plan for plan in object_plans
        if _plan_role(plan) in {"beat_orb", "beat_pad"} or _plan_object_id(plan) in {"35", "36", "84", "140", "141", "1022"}
    ]
    xs = sorted(x for plan in input_events if (x := _plan_x(plan)) is not None)
    if len(xs) < 4:
        return []
        
    window = 180.0
    allowed = 5 if _difficulty_relief(difficulty) > 0.75 else 7
    warnings: list[PlayabilityWarning] = []
    
    # Sliding window optimization O(n)
    left = 0
    for right in range(len(xs)):
        while xs[right] - xs[left] > window:
            left += 1
        
        count = right - left + 1
        if count > allowed:
            warnings.append(
                PlayabilityWarning(
                    warning_type="excessive_input_density",
                    severity="warning",
                    time=None,
                    x=xs[left],
                    mode="unknown",
                    speed_state="unknown",
                    message=f"{count} input events within {window}px",
                )
            )
            # Break early after finding one violation in this area to avoid spamming
            # Move left forward to start of next possible window
            left = right + 1 
            if len(warnings) >= 32:
                break
                
    return warnings


def validate_orb_pad_hazard_spacing(
    object_plans: list[ObjectPlan | dict],
    *,
    difficulty: str | float | int,
) -> list[PlayabilityWarning]:
    warnings: list[PlayabilityWarning] = []
    relief = _difficulty_relief(difficulty)
    inputs = [
        plan for plan in object_plans
        if _plan_role(plan) in {"beat_orb", "beat_pad"} or _plan_object_id(plan) in {"35", "36", "84", "140", "141", "1022"}
    ]
    hazards = [plan for plan in object_plans if _plan_role(plan) == "hazard" or _plan_object_id(plan) in {"8", "39"}]
    
    if not inputs or not hazards:
        return []
        
    hazard_xs = sorted(x for hazard in hazards if (x := _plan_x(hazard)) is not None)
    min_gap = 72.0 * relief
    
    import bisect
    for input_plan in inputs:
        input_x = _plan_x(input_plan)
        if input_x is None:
            continue
            
        # Find hazard right after input_x
        idx = bisect.bisect_left(hazard_xs, input_x)
        if idx < len(hazard_xs):
            hazard_x = hazard_xs[idx]
            if input_x < hazard_x < input_x + min_gap:
                warnings.append(
                    PlayabilityWarning(
                        warning_type="orb_pad_hazard_spacing",
                        severity="warning",
                        time=None,
                        x=input_x,
                        mode="unknown",
                        speed_state="unknown",
                        message=f"hazard is within {min_gap:.2f}px after orb/pad",
                    )
                )
                if len(warnings) >= 32:
                    break
    return warnings


def validate_mode_transition_safety(
    object_plans: list[ObjectPlan | dict],
    section_plans: list[SectionPlan],
    *,
    difficulty: str | float | int,
) -> list[PlayabilityWarning]:
    # Mode portals share the same conservative margin as speed portals in v1.
    return validate_portal_safety_margin(
        object_plans,
        section_plans,
        difficulty=difficulty,
    )


def validate_playability_v1(
    events: list[object] | None = None,
    object_plans: list[ObjectPlan | dict] | None = None,
    section_plans: list[SectionPlan] | None = None,
    speed_objects: list[object] | None = None,
    difficulty: str | float | int = "normal",
) -> list[PlayabilityWarning]:
    plans = list(object_plans or [])
    plans.extend(event for event in (events or []) if isinstance(event, (ObjectPlan, dict)))
    sections = section_plans or []
    warnings: list[PlayabilityWarning] = []
    warnings.extend(validate_min_event_spacing(plans, sections, difficulty=difficulty))
    warnings.extend(validate_portal_safety_margin(plans, sections, difficulty=difficulty))
    warnings.extend(validate_orb_pad_hazard_spacing(plans, difficulty=difficulty))
    warnings.extend(validate_excessive_input_density(plans, difficulty=difficulty))
    return warnings[:64]


def parse_save_string_safe(level_string: str) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    try:
        if "<k>" in level_string or "<plist" in level_string or "<d>" in level_string:
            tags = parse_gmd_text(level_string)
            encoded = tags.get("k4", ("", ""))[1]
            decoded = decode_level_data(encoded)
        else:
            decoded = level_string
        return split_level_objects(decoded), issues
    except Exception as exc:  # noqa: BLE001
        return [], [f"parse_save_string_failed: {exc}"]


def validate_encoder_safe_keys(
    objects: list[str],
    *,
    safe_mode: bool,
    max_group_id: int | None = None,
) -> list[str]:
    issues: list[str] = []
    for obj in objects:
        obj_id = extract_object_id(obj)
        if obj_id is None:
            continue
        for key, value in zip(obj.split(",")[::2], obj.split(",")[1::2]):
            if value.lower() in {"nan", "inf", "-inf"}:
                issues.append(f"invalid_numeric_value: {key}={value}")
        if obj_id in _TRIGGER_IDS and obj_id not in _SPEED_PORTAL_IDS:
            trigger_type = _TRIGGER_TYPE_BY_ID.get(obj_id)
            if trigger_type is None:
                issues.append(f"unsupported_trigger: {obj_id}")
            elif not is_trigger_allowed_in_mode(trigger_type, "safe" if safe_mode else "advanced"):
                issues.append(f"unsupported_trigger_in_mode: {trigger_type}")
    issues.extend(_check_group_bounds(objects, max_group_id=max_group_id))
    return issues


def round_trip_validate(
    level_string: str,
    *,
    safe_mode: bool = False,
    max_group_id: int | None = None,
) -> dict[str, Any]:
    objects, issues = parse_save_string_safe(level_string)
    if issues:
        return {
            "valid": False,
            "parse_ok": False,
            "encode_ok": False,
            "object_count_before": 0,
            "object_count_after": 0,
            "issues": issues,
        }
    issues.extend(validate_encoder_safe_keys(objects, safe_mode=safe_mode, max_group_id=max_group_id))
    try:
        header = extract_level_header(level_string) or "kA11,0"
        decoded = ";".join([header, *objects]) + ";"
        encoded = encode_level_data(decoded)
        recovered = split_level_objects(decode_level_data(encoded))
    except Exception as exc:  # noqa: BLE001
        issues.append(f"round_trip_encode_failed: {exc}")
        recovered = []
    if len(recovered) != len(objects):
        issues.append(f"round_trip_count_mismatch: {len(objects)} != {len(recovered)}")
    return {
        "valid": not issues,
        "parse_ok": True,
        "encode_ok": not any("encode_failed" in issue for issue in issues),
        "object_count_before": len(objects),
        "object_count_after": len(recovered),
        "issues": issues,
    }


def _check_k95_sync(tags: dict, actual_object_count: int) -> list[str]:
    issues: list[str] = []
    k95_entry = tags.get("k95")
    if k95_entry:
        _, k95_value = k95_entry
        try:
            declared = int(k95_value)
            if declared != actual_object_count:
                issues.append(
                    f"k95_mismatch: k95={declared} but actual object count={actual_object_count}"
                )
        except (ValueError, TypeError):
            issues.append(f"k95_invalid: k95 value cannot be parsed as int: {k95_value!r}")
    return issues


def validate_gmd_text(
    raw_text: str,
    *,
    check_structure: bool = True,
    object_budget: int | None = None,
    max_group_id: int | None = None,
    safe_mode: bool = False,
    check_playability: bool = False,
    check_round_trip: bool = False,
) -> tuple[bool, list[str]]:
    issues: list[str] = []

    try:
        tags = parse_gmd_text(raw_text)
    except Exception as exc:  # noqa: BLE001
        return False, [f"parse failed: {exc}"]

    if "k2" not in tags:
        issues.append("missing k2 (level name)")
    if "k4" not in tags:
        issues.append("missing k4 (level data)")
    else:
        _, encoded_level_data = tags["k4"]
        try:
            decoded = decode_level_data(encoded_level_data)
            if not decoded.strip():
                issues.append("decoded level data is empty")
            elif not extract_level_header(decoded):
                issues.append("missing level header section in decoded level data")
            else:
                level_objects = split_level_objects(decoded)
                visible_objects = [
                    obj for obj in level_objects if extract_object_id(obj) is not None
                ]
                if not visible_objects:
                    issues.append("decoded level data contains no objects")
                elif check_structure:
                    issues.extend(_check_x_monotone(visible_objects))
                    issues.extend(_check_speed_portals_sorted(visible_objects))
                    issues.extend(_check_orphan_triggers(visible_objects))
                    issues.extend(_check_trigger_durations(visible_objects))
                    issues.extend(_check_spawn_delays(visible_objects))
                    issues.extend(_check_group_bounds(visible_objects, max_group_id=max_group_id))
                    issues.extend(_check_trigger_schema(visible_objects, safe_mode=safe_mode, max_group_id=max_group_id))
                    issues.extend(_check_object_budget(visible_objects, object_budget=object_budget))
                    issues.extend(_check_safe_mode_triggers(visible_objects, safe_mode=safe_mode))
                    if check_playability:
                        issues.extend(_check_tight_spacing(visible_objects))
                    issues.extend(_check_k95_sync(tags, len(visible_objects)))
                    if check_round_trip:
                        round_trip = round_trip_validate(
                            decoded,
                            safe_mode=safe_mode,
                            max_group_id=max_group_id,
                        )
                        issues.extend(str(issue) for issue in round_trip["issues"])
        except Exception as exc:  # noqa: BLE001
            issues.append(f"k4 decode failed: {exc}")

    return len(issues) == 0, issues


def validate_gmd_file(
    path: Path,
    *,
    check_structure: bool = True,
    object_budget: int | None = None,
    max_group_id: int | None = None,
    safe_mode: bool = False,
    check_playability: bool = False,
    check_round_trip: bool = False,
) -> tuple[bool, list[str]]:
    raw_text = path.read_text(encoding="utf-8")
    return validate_gmd_text(
        raw_text,
        check_structure=check_structure,
        object_budget=object_budget,
        max_group_id=max_group_id,
        safe_mode=safe_mode,
        check_playability=check_playability,
        check_round_trip=check_round_trip,
    )
