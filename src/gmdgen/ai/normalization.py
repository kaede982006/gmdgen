from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from gmdgen.gd.plans import TriggerMode
from gmdgen.gd.triggers import get_trigger_schema


_CANONICAL_OBJECT_ROLES: tuple[str, ...] = (
    "ground_or_structure",
    "structure",
    "structure_accent",
    "beat_orb",
    "beat_pad",
    "visual_accent_target",
    "decoration",
    "safe_decoration",
    "section_transition_structure",
    "ai_structure",
    "ai_decoration",
    "ai_orb",
    "ai_pad",
)

_ROLE_ALIASES: dict[str, str] = {
    "orb": "beat_orb",
    "jump_orb": "beat_orb",
    "gameplay_orb": "beat_orb",
    "ai_gameplay_orb": "beat_orb",
    "jump_pad": "beat_pad",
    "pad": "beat_pad",
    "gameplay_pad": "beat_pad",
    "jump-pad": "beat_pad",
    "jump pad": "beat_pad",
    "block": "ai_structure",
    "platform": "ai_structure",
    "hazard": "structure_accent",
    "spike": "structure_accent",
    "accent": "visual_accent_target",
    "glow": "safe_decoration",
    "deco": "safe_decoration",
}

_EASING_ALIASES: dict[str, str] = {
    "": "linear",
    "none": "linear",
    "linear": "linear",
    "easein": "ease_in",
    "ease-in": "ease_in",
    "ease_in": "ease_in",
    "in": "ease_in",
    "easeout": "ease_out",
    "ease-out": "ease_out",
    "ease_out": "ease_out",
    "out": "ease_out",
    "easeinout": "ease_in_out",
    "ease-in-out": "ease_in_out",
    "ease_in_out": "ease_in_out",
    "in_out": "ease_in_out",
    "in-out": "ease_in_out",
}

_TRIGGER_TOP_LEVEL_FIELDS: frozenset[str] = frozenset(
    {
        "target_group",
        "secondary_group",
        "duration",
        "easing",
        "spawn_delay",
        "multi_trigger",
        "editor_disable",
    }
)

_TRIGGER_INTENT_FIELDS: frozenset[str] = frozenset(
    {
        "trigger_kind",
        "purpose",
        "target_role",
        "intensity",
        "duration_hint",
        "section_id",
    }
)

_GENERIC_TRIGGER_HINT_FIELDS: frozenset[str] = frozenset(
    {
        "move_x",
        "move_y",
        "opacity",
        "color_channel",
        "copy_color_channel",
        "exclusive",
    }
)


@dataclass(slots=True)
class AIPlanNormalizationReport:
    warnings: list[str] = field(default_factory=list)
    pruned_trigger_property_count: int = 0
    ignored_irrelevant_trigger_property_count: int = 0
    normalized_object_role_count: int = 0
    normalized_easing_count: int = 0
    lifted_trigger_property_count: int = 0
    materialized_trigger_intent_count: int = 0
    auto_assigned_target_group_count: int = 0
    unresolved_missing_target_group_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "warnings": list(self.warnings),
            "pruned_trigger_property_count": self.pruned_trigger_property_count,
            "ignored_irrelevant_trigger_property_count": self.ignored_irrelevant_trigger_property_count,
            "normalized_object_role_count": self.normalized_object_role_count,
            "normalized_easing_count": self.normalized_easing_count,
            "lifted_trigger_property_count": self.lifted_trigger_property_count,
            "materialized_trigger_intent_count": self.materialized_trigger_intent_count,
            "auto_assigned_target_group_count": self.auto_assigned_target_group_count,
            "unresolved_missing_target_group_count": self.unresolved_missing_target_group_count,
        }


@dataclass(slots=True)
class TriggerIntent:
    trigger_kind: str
    purpose: str = "beat_accent"
    target_role: str = "section_group"
    intensity: float = 0.5
    duration_hint: float | None = None
    beat_aligned_time: float | None = None
    section_id: int | None = None


def allowed_object_roles() -> tuple[str, ...]:
    return _CANONICAL_OBJECT_ROLES


def normalize_object_role(role: Any) -> tuple[str, bool]:
    raw = str(role or "").strip()
    lowered = raw.lower().replace("-", "_").replace(" ", "_")
    canonical = _ROLE_ALIASES.get(lowered, lowered)
    if canonical in _CANONICAL_OBJECT_ROLES:
        return canonical, canonical != raw
    return raw, False


def validate_object_role(role: Any) -> bool:
    normalized, _changed = normalize_object_role(role)
    return normalized in _CANONICAL_OBJECT_ROLES


def normalize_easing(value: Any) -> tuple[str | None, bool]:
    if value is None:
        return None, False
    raw = str(value).strip()
    key = raw.replace(" ", "").lower()
    normalized = _EASING_ALIASES.get(key)
    if normalized is None:
        return "linear", raw != "linear"
    return normalized, normalized != raw


def normalize_ai_level_plan_response_payload(
    payload: dict[str, Any],
    *,
    safe_mode: bool = True,
    max_group_id: int = 9999,
    section_plans: list[Any] | None = None,
) -> tuple[dict[str, Any], AIPlanNormalizationReport]:
    normalized = deepcopy(payload)
    report = AIPlanNormalizationReport()
    normalize_ai_object_plans(normalized.get("object_plans", []), report)
    materialize_ai_trigger_contract(
        normalized,
        report,
        max_group_id=max_group_id,
        section_plans=section_plans or [],
    )
    normalize_ai_trigger_plans(
        normalized.get("trigger_plans", []),
        report,
        mode=TriggerMode.SAFE if safe_mode else TriggerMode.ADVANCED,
    )
    return normalized, report


def materialize_ai_trigger_contract(
    payload: dict[str, Any],
    report: AIPlanNormalizationReport,
    *,
    max_group_id: int = 9999,
    section_plans: list[Any] | None = None,
) -> None:
    """Turn model trigger hints into safer low-level trigger fields before validation.

    The model is allowed to provide intent-like metadata inside the nullable
    `properties` object. This pass consumes those hints, assigns groups, and
    writes only schema-owned fields before the strict validator sees the plan.
    """

    object_plans = payload.get("object_plans", [])
    trigger_plans = payload.get("trigger_plans", [])
    if not isinstance(object_plans, list) or not isinstance(trigger_plans, list):
        return

    allocator = _RawGroupAllocator(object_plans, section_plans or [], max_group_id=max_group_id)
    for idx, plan in enumerate(trigger_plans):
        if not isinstance(plan, dict):
            continue
        properties = plan.get("properties")
        if properties is None:
            properties = {}
            plan["properties"] = properties
        if not isinstance(properties, dict):
            properties = {}
            plan["properties"] = properties

        intent = _consume_trigger_intent(plan, properties, idx, report)
        if intent.trigger_kind:
            schema = get_trigger_schema(intent.trigger_kind)
            if schema is not None:
                plan["trigger_type"] = intent.trigger_kind
                plan["object_id"] = schema.object_id

        trigger_type = str(plan.get("trigger_type", "")).strip().lower()
        schema = get_trigger_schema(trigger_type)
        if schema is None:
            continue

        _apply_materializer_defaults(plan, schema, intent)
        requires_target = "target_group" in {prop.name for prop in schema.properties if prop.required}
        current_target = _optional_int(plan.get("target_group"))
        if current_target is not None:
            if allocator.ensure_group_exists(current_target, plan):
                report.auto_assigned_target_group_count += 1
                report.warnings.append(f"repaired_orphan_target_group[{idx}]: {current_target}")
            continue
        if not requires_target and not intent.target_role:
            continue

        target_group = allocator.group_for_trigger(plan, intent)
        if target_group is None:
            if requires_target:
                report.unresolved_missing_target_group_count += 1
                report.warnings.append(f"unresolved_missing_target_group[{idx}]: {trigger_type}")
            continue
        plan["target_group"] = target_group
        report.auto_assigned_target_group_count += 1
        report.warnings.append(
            f"auto_assigned_target_group[{idx}]: {trigger_type}->{target_group}"
        )


def normalize_ai_object_plans(
    object_plans: Any,
    report: AIPlanNormalizationReport,
) -> None:
    if not isinstance(object_plans, list):
        return
    for idx, plan in enumerate(object_plans):
        if not isinstance(plan, dict):
            continue
        raw_role = plan.get("role", "")
        role, changed = normalize_object_role(raw_role)
        if changed:
            plan["role"] = role
            report.normalized_object_role_count += 1
            report.warnings.append(f"normalized_object_role[{idx}]: {raw_role} -> {role}")


def normalize_ai_trigger_plans(
    trigger_plans: Any,
    report: AIPlanNormalizationReport,
    *,
    mode: TriggerMode = TriggerMode.SAFE,
) -> None:
    if not isinstance(trigger_plans, list):
        return
    for idx, plan in enumerate(trigger_plans):
        if not isinstance(plan, dict):
            continue
        trigger_type = str(plan.get("trigger_type", "")).strip().lower()
        plan["trigger_type"] = trigger_type
        schema = get_trigger_schema(trigger_type)
        if schema is None:
            continue
        properties = plan.get("properties")
        if properties is None:
            properties = {}
            plan["properties"] = properties
        if not isinstance(properties, dict):
            plan["properties"] = {}
            properties = plan["properties"]
            report.warnings.append(f"trigger_properties_replaced[{idx}]: expected object")

        _lift_top_level_trigger_properties(plan, properties, idx, report)
        _normalize_trigger_easing(plan, schema, idx, report)
        _prune_irrelevant_trigger_properties(plan, schema, idx, report)


def _lift_top_level_trigger_properties(
    plan: dict[str, Any],
    properties: dict[str, Any],
    idx: int,
    report: AIPlanNormalizationReport,
) -> None:
    for name in sorted(set(properties) & _TRIGGER_TOP_LEVEL_FIELDS):
        value = properties.pop(name)
        if plan.get(name) is None or plan.get(name) == "":
            plan[name] = value
            report.lifted_trigger_property_count += 1
            report.warnings.append(f"lifted_trigger_property[{idx}]: properties.{name} -> {name}")
        else:
            report.pruned_trigger_property_count += 1
            report.warnings.append(f"pruned_duplicate_trigger_property[{idx}]: properties.{name}")


def _normalize_trigger_easing(
    plan: dict[str, Any],
    schema: Any,
    idx: int,
    report: AIPlanNormalizationReport,
) -> None:
    easing_supported = "easing" in {prop.name for prop in schema.properties}
    raw_easing = plan.get("easing")
    if not easing_supported:
        if raw_easing not in (None, ""):
            plan["easing"] = None
            report.pruned_trigger_property_count += 1
            report.warnings.append(f"pruned_irrelevant_trigger_property[{idx}]: {schema.trigger_type}.easing")
        return
    normalized, changed = normalize_easing(raw_easing)
    if raw_easing is not None and changed:
        report.normalized_easing_count += 1
        report.warnings.append(f"normalized_easing[{idx}]: {raw_easing} -> {normalized}")
    plan["easing"] = normalized or "linear"


def _prune_irrelevant_trigger_properties(
    plan: dict[str, Any],
    schema: Any,
    idx: int,
    report: AIPlanNormalizationReport,
) -> None:
    properties = plan.get("properties")
    if not isinstance(properties, dict):
        return
    allowed = {prop.name for prop in schema.properties}
    for name in sorted(list(properties.keys())):
        if name not in allowed:
            if name in _GENERIC_TRIGGER_HINT_FIELDS:
                properties.pop(name, None)
                report.ignored_irrelevant_trigger_property_count += 1
                report.warnings.append(
                    f"ignored_irrelevant_trigger_hint[{idx}]: {schema.trigger_type}.{name}"
                )
                continue
            properties.pop(name, None)
            report.pruned_trigger_property_count += 1
            report.warnings.append(
                f"pruned_irrelevant_trigger_property[{idx}]: {schema.trigger_type}.{name}"
            )


def _consume_trigger_intent(
    plan: dict[str, Any],
    properties: dict[str, Any],
    idx: int,
    report: AIPlanNormalizationReport,
) -> TriggerIntent:
    trigger_kind = str(properties.pop("trigger_kind", "") or plan.get("trigger_type", "") or "").strip().lower()
    purpose = str(properties.pop("purpose", "") or "beat_accent").strip().lower()
    target_role = str(properties.pop("target_role", "") or _target_role_from_purpose(purpose)).strip().lower()
    intensity = _clamp_float(properties.pop("intensity", 0.5), 0.0, 1.0, 0.5)
    duration_hint = _optional_float(properties.pop("duration_hint", None))
    section_id = _optional_int(properties.pop("section_id", None))
    if section_id is None:
        safety = plan.get("safety_flags")
        if isinstance(safety, dict):
            section_id = _optional_int(safety.get("section_id"))
    if trigger_kind or purpose != "beat_accent" or target_role:
        report.materialized_trigger_intent_count += 1
        report.warnings.append(
            f"materialized_trigger_intent[{idx}]: kind={trigger_kind or plan.get('trigger_type')} purpose={purpose}"
        )
    return TriggerIntent(
        trigger_kind=trigger_kind,
        purpose=purpose,
        target_role=target_role or "section_group",
        intensity=intensity,
        duration_hint=duration_hint,
        beat_aligned_time=_optional_float(plan.get("beat_aligned_time")),
        section_id=section_id,
    )


def _apply_materializer_defaults(plan: dict[str, Any], schema: Any, intent: TriggerIntent) -> None:
    if plan.get("object_id") in (None, "", "0", 0):
        plan["object_id"] = schema.object_id
    elif str(plan.get("object_id")) != str(schema.object_id):
        plan["object_id"] = schema.object_id

    props = plan.get("properties")
    if not isinstance(props, dict):
        props = {}
        plan["properties"] = props

    duration = _optional_float(plan.get("duration"))
    if (duration is None or duration <= 0.0) and intent.duration_hint is not None:
        plan["duration"] = max(0.02, min(2.0, intent.duration_hint))
    elif duration is None:
        for prop in schema.properties:
            if prop.name == "duration" and prop.default is not None:
                plan["duration"] = prop.default
                break

    if schema.trigger_type == "move":
        property_easing = props.pop("easing", None)
        if plan.get("easing") in (None, "") and property_easing not in (None, ""):
            plan["easing"] = property_easing
        plan["easing"] = plan.get("easing") or "linear"
        if props.get("move_x") is None and props.get("move_y") is None:
            props["move_y" if intent.purpose in {"beat_accent", "drop_accent"} else "move_x"] = round(8 + 24 * intent.intensity, 3)
    elif schema.trigger_type in {"pulse", "color"}:
        if props.get("color_channel") is None:
            props["color_channel"] = 1
    elif schema.trigger_type == "alpha":
        if props.get("opacity") is None:
            props["opacity"] = round(max(0.15, min(1.0, 0.35 + intent.intensity * 0.55)), 3)

    if schema.trigger_type == "spawn" and plan.get("spawn_delay") is None:
        plan["spawn_delay"] = 0.0


class _RawGroupAllocator:
    def __init__(self, object_plans: list[Any], section_plans: list[Any], *, max_group_id: int) -> None:
        self.object_plans = [plan for plan in object_plans if isinstance(plan, dict)]
        self.section_plans = section_plans
        self.max_group_id = max(1, int(max_group_id or 9999))
        existing = [
            int(group_id)
            for plan in self.object_plans
            for group_id in (plan.get("group_ids") or [])
            if isinstance(group_id, int) and group_id > 0
        ]
        self._next_id = min(self.max_group_id, max(existing, default=0) + 1)

    def group_for_trigger(self, trigger: dict[str, Any], intent: TriggerIntent) -> int | None:
        section_id = intent.section_id
        if section_id is None:
            section_id = self._section_id_for_x(_optional_float(trigger.get("x")))
        objects = self._objects_for(section_id, intent.target_role, intent.purpose)
        if not objects and self.object_plans:
            objects = self._objects_for(section_id, "section_group", intent.purpose)
        if not objects:
            return None
        existing = self._first_group(objects)
        if existing is not None:
            return existing
        group_id = self._allocate()
        if group_id is None:
            return None
        for obj in objects:
            groups = obj.setdefault("group_ids", [])
            if isinstance(groups, list) and group_id not in groups:
                groups.append(group_id)
        return group_id

    def ensure_group_exists(self, group_id: int, trigger: dict[str, Any]) -> bool:
        if any(group_id in (obj.get("group_ids") or []) for obj in self.object_plans):
            return False
        section_id = self._section_id_for_x(_optional_float(trigger.get("x")))
        objects = self._objects_for(section_id, "section_group", "beat_accent")
        if not objects:
            return False
        for obj in objects:
            groups = obj.setdefault("group_ids", [])
            if isinstance(groups, list) and group_id not in groups:
                groups.append(group_id)
                return True
        return False

    def _allocate(self) -> int | None:
        if self._next_id > self.max_group_id:
            return None
        group_id = self._next_id
        self._next_id += 1
        return group_id

    def _objects_for(self, section_id: int | None, target_role: str, purpose: str) -> list[dict[str, Any]]:
        candidates = [
            obj
            for obj in self.object_plans
            if section_id is None or self._object_section_id(obj) == section_id
        ]
        if not candidates:
            candidates = list(self.object_plans)
        role = str(target_role or "").lower()
        if role in {"decoration_group", "background_group"} or purpose in {"drop_accent", "decoration", "visibility"}:
            filtered = [obj for obj in candidates if _is_decoration_role(obj.get("role"))]
            return filtered or candidates[:1]
        if role == "gameplay_group":
            filtered = [obj for obj in candidates if _is_gameplay_role(obj.get("role"))]
            return filtered or candidates[:1]
        return candidates[: max(1, min(4, len(candidates)))]

    def _first_group(self, objects: list[dict[str, Any]]) -> int | None:
        for obj in objects:
            for group_id in obj.get("group_ids") or []:
                if isinstance(group_id, int) and 1 <= group_id <= self.max_group_id:
                    return group_id
        return None

    def _object_section_id(self, obj: dict[str, Any]) -> int | None:
        flags = obj.get("safety_flags")
        if isinstance(flags, dict):
            value = _optional_int(flags.get("section_id"))
            if value is not None:
                return value
        return self._section_id_for_x(_optional_float(obj.get("x")))

    def _section_id_for_x(self, x_value: float | None) -> int | None:
        if x_value is None or not self.section_plans:
            return None
        best_idx = 0
        best_dist = float("inf")
        for idx, section in enumerate(self.section_plans):
            start_x = float(getattr(section, "start_x", 0.0))
            end_x = float(getattr(section, "end_x", start_x))
            if start_x <= x_value <= end_x:
                return idx
            dist = min(abs(x_value - start_x), abs(x_value - end_x))
            if dist < best_dist:
                best_idx, best_dist = idx, dist
        return best_idx


def _target_role_from_purpose(purpose: str) -> str:
    if purpose in {"drop_accent", "decoration", "visibility"}:
        return "decoration_group"
    if purpose in {"gameplay", "beat_accent"}:
        return "gameplay_group"
    return "section_group"


def _is_decoration_role(role: Any) -> bool:
    text = str(role or "").lower()
    return "decor" in text or "accent" in text or "visual" in text or "background" in text


def _is_gameplay_role(role: Any) -> bool:
    text = str(role or "").lower()
    return "orb" in text or "pad" in text or "gameplay" in text or "structure" in text


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _clamp_float(value: Any, low: float, high: float, default: float) -> float:
    parsed = _optional_float(value)
    if parsed is None:
        parsed = default
    return max(low, min(high, parsed))
