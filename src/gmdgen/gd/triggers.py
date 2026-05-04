from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gmdgen.gd.plans import TriggerMode, TriggerPlan, normalize_trigger_mode


@dataclass(slots=True, frozen=True)
class TriggerProperty:
    name: str
    save_key: str | None
    value_type: str
    required: bool = False
    default: Any = None
    min_value: float | None = None
    max_value: float | None = None
    enum_values: tuple[str, ...] = ()
    safe_mode_allowed: bool = True
    advanced_mode_allowed: bool = True
    description: str = ""


@dataclass(slots=True, frozen=True)
class TriggerSchemaV2:
    trigger_type: str
    object_id: str
    properties: tuple[TriggerProperty, ...] = ()
    target_group_properties: tuple[str, ...] = ()
    duration_properties: tuple[str, ...] = ()
    color_properties: tuple[str, ...] = ()
    easing_properties: tuple[str, ...] = ()
    spawn_delay_properties: tuple[str, ...] = ()
    safe_mode_allowed: bool = False
    advanced_mode_allowed: bool = True
    unsupported_reason: str = ""

    @property
    def required_properties(self) -> tuple[str, ...]:
        return tuple(prop.name for prop in self.properties if prop.required)


def _prop(
    name: str,
    key: str | None,
    value_type: str,
    *,
    required: bool = False,
    default: Any = None,
    min_value: float | None = None,
    max_value: float | None = None,
    enum_values: tuple[str, ...] = (),
    safe: bool = True,
    advanced: bool = True,
    description: str = "",
) -> TriggerProperty:
    return TriggerProperty(
        name=name,
        save_key=key,
        value_type=value_type,
        required=required,
        default=default,
        min_value=min_value,
        max_value=max_value,
        enum_values=enum_values,
        safe_mode_allowed=safe,
        advanced_mode_allowed=advanced,
        description=description,
    )


_TARGET = _prop("target_group", "51", "int", required=True, min_value=1, max_value=9999)
_SECONDARY = _prop("secondary_group", "71", "int", min_value=1, max_value=9999)
_DURATION = _prop("duration", "10", "float", default=0.2, min_value=0.0, max_value=8.0)
_SHORT_DURATION = _prop("duration", "10", "float", default=0.18, min_value=0.02, max_value=2.0)
_SPAWN_DELAY = _prop("spawn_delay", "63", "float", default=0.0, min_value=0.0, max_value=8.0)
_MULTI = _prop("multi_trigger", "35", "bool", default=False)
_EDITOR_DISABLE = _prop("editor_disable", "58", "bool", default=False)
_EASING = _prop(
    "easing",
    None,
    "enum",
    default="linear",
    enum_values=("linear", "ease_in", "ease_out", "ease_in_out"),
    safe=False,
    description="Reserved until a version-specific GD 2.2 easing key is proven.",
)
_MOVE_X = _prop("move_x", None, "float", default=0.0, min_value=-900.0, max_value=900.0, safe=False)
_MOVE_Y = _prop("move_y", None, "float", default=0.0, min_value=-900.0, max_value=900.0, safe=False)
_COLOR_CHANNEL = _prop("color_channel", None, "int", default=1, min_value=1, max_value=999, safe=False)
_OPACITY = _prop("opacity", None, "float", default=1.0, min_value=0.0, max_value=1.0, safe=False)


TRIGGER_SCHEMA_REGISTRY: dict[str, TriggerSchemaV2] = {
    "move": TriggerSchemaV2(
        trigger_type="move",
        object_id="901",
        properties=(_TARGET, _DURATION, _MOVE_X, _MOVE_Y, _EASING, _MULTI, _EDITOR_DISABLE),
        target_group_properties=("target_group",),
        duration_properties=("duration",),
        easing_properties=("easing",),
        safe_mode_allowed=True,
    ),
    "pulse": TriggerSchemaV2(
        trigger_type="pulse",
        object_id="1006",
        properties=(_TARGET, _SHORT_DURATION, _COLOR_CHANNEL, _MULTI, _EDITOR_DISABLE),
        target_group_properties=("target_group",),
        duration_properties=("duration",),
        color_properties=("color_channel",),
        safe_mode_allowed=True,
    ),
    "alpha": TriggerSchemaV2(
        trigger_type="alpha",
        object_id="1007",
        properties=(_TARGET, _SHORT_DURATION, _OPACITY, _MULTI, _EDITOR_DISABLE),
        target_group_properties=("target_group",),
        duration_properties=("duration",),
        safe_mode_allowed=True,
    ),
    "spawn": TriggerSchemaV2(
        trigger_type="spawn",
        object_id="1268",
        properties=(_TARGET, _SPAWN_DELAY, _MULTI, _EDITOR_DISABLE),
        target_group_properties=("target_group",),
        spawn_delay_properties=("spawn_delay",),
        safe_mode_allowed=True,
    ),
    "stop": TriggerSchemaV2(
        trigger_type="stop",
        object_id="899",
        properties=(_TARGET, _EDITOR_DISABLE),
        target_group_properties=("target_group",),
        safe_mode_allowed=True,
    ),
    "color": TriggerSchemaV2(
        trigger_type="color",
        object_id="29",
        properties=(_DURATION, _COLOR_CHANNEL, _TARGET, _MULTI, _EDITOR_DISABLE),
        target_group_properties=("target_group",),
        duration_properties=("duration",),
        color_properties=("color_channel",),
        safe_mode_allowed=True,
    ),
    "toggle": TriggerSchemaV2(
        trigger_type="toggle",
        object_id="33",
        properties=(_TARGET, _MULTI, _EDITOR_DISABLE),
        target_group_properties=("target_group",),
        safe_mode_allowed=True,
    ),
    "follow": TriggerSchemaV2(
        trigger_type="follow",
        object_id="1347",
        properties=(_TARGET, _SECONDARY, _DURATION, _EDITOR_DISABLE),
        target_group_properties=("target_group", "secondary_group"),
        duration_properties=("duration",),
        safe_mode_allowed=False,
    ),
    "shake": TriggerSchemaV2(
        trigger_type="shake",
        object_id="1520",
        properties=(_SHORT_DURATION, _EDITOR_DISABLE),
        duration_properties=("duration",),
        safe_mode_allowed=False,
    ),
    "count": TriggerSchemaV2(
        trigger_type="count",
        object_id="1611",
        properties=(_TARGET, _EDITOR_DISABLE),
        target_group_properties=("target_group",),
        safe_mode_allowed=False,
        unsupported_reason="Count trigger semantics need full GD 2.2 key map before safe encoding.",
    ),
    "collision": TriggerSchemaV2(
        trigger_type="collision",
        object_id="1815",
        properties=(_TARGET, _EDITOR_DISABLE),
        target_group_properties=("target_group",),
        safe_mode_allowed=False,
        unsupported_reason="Collision block/property mapping is intentionally advanced-only.",
    ),
    "pickup": TriggerSchemaV2(
        trigger_type="pickup",
        object_id="1817",
        properties=(_TARGET, _EDITOR_DISABLE),
        target_group_properties=("target_group",),
        safe_mode_allowed=False,
        unsupported_reason="Pickup trigger key map is incomplete.",
    ),
}


def get_trigger_schema(trigger_type: str) -> TriggerSchemaV2 | None:
    return TRIGGER_SCHEMA_REGISTRY.get(str(trigger_type).strip().lower())


def list_supported_triggers(mode: TriggerMode | str | bool | None = TriggerMode.SAFE) -> list[str]:
    normalized = normalize_trigger_mode(mode)
    result = []
    for trigger_type, schema in sorted(TRIGGER_SCHEMA_REGISTRY.items()):
        if normalized == TriggerMode.SAFE and schema.safe_mode_allowed:
            result.append(trigger_type)
        elif normalized == TriggerMode.ADVANCED and schema.advanced_mode_allowed:
            result.append(trigger_type)
    return result


def reject_unsupported_trigger(
    trigger_plan: TriggerPlan,
    mode: TriggerMode | str | bool | None = TriggerMode.SAFE,
) -> str | None:
    schema = get_trigger_schema(trigger_plan.trigger_type)
    if schema is None:
        return f"unsupported_trigger_schema: {trigger_plan.trigger_type}"
    normalized = normalize_trigger_mode(mode)
    if normalized == TriggerMode.SAFE and not schema.safe_mode_allowed:
        return f"trigger_not_allowed_in_safe_mode: {trigger_plan.trigger_type}"
    if normalized == TriggerMode.ADVANCED and not schema.advanced_mode_allowed:
        return f"trigger_not_allowed_in_advanced_mode: {trigger_plan.trigger_type}"
    return None


def validate_trigger_properties(
    trigger_plan: TriggerPlan,
    mode: TriggerMode | str | bool | None = TriggerMode.SAFE,
    *,
    max_group_id: int | None = None,
) -> list[str]:
    rejection = reject_unsupported_trigger(trigger_plan, mode)
    if rejection:
        return [rejection]
    schema = get_trigger_schema(trigger_plan.trigger_type)
    assert schema is not None
    issues: list[str] = []
    if str(trigger_plan.object_id) != schema.object_id:
        issues.append(f"trigger_object_id_mismatch: {trigger_plan.trigger_type}")
    known = {prop.name for prop in schema.properties}
    extras = set(getattr(trigger_plan, "properties", {}).keys())
    for name in sorted(extras - known):
        issues.append(f"unknown_trigger_property: {trigger_plan.trigger_type}.{name}")
    for prop in schema.properties:
        value = _property_value(trigger_plan, prop.name)
        if prop.required and value is None:
            issues.append(f"missing_trigger_property: {trigger_plan.trigger_type}.{prop.name}")
            continue
        if value is None:
            continue
        if prop.value_type == "enum" and prop.enum_values and str(value) not in prop.enum_values:
            issues.append(f"invalid_trigger_enum: {trigger_plan.trigger_type}.{prop.name}")
        if prop.min_value is not None and float(value) < prop.min_value:
            issues.append(f"trigger_property_below_min: {trigger_plan.trigger_type}.{prop.name}")
        if prop.max_value is not None and float(value) > prop.max_value:
            issues.append(f"trigger_property_above_max: {trigger_plan.trigger_type}.{prop.name}")
        if prop.name in {"target_group", "secondary_group"} and max_group_id is not None:
            if not (1 <= int(value) <= max_group_id):
                issues.append(f"trigger_group_bounds: {trigger_plan.trigger_type}.{prop.name}")
    return issues


def apply_trigger_property_defaults(
    trigger_plan: TriggerPlan,
    mode: TriggerMode | str | bool | None = TriggerMode.SAFE,
) -> TriggerPlan | None:
    if reject_unsupported_trigger(trigger_plan, mode):
        return None
    schema = get_trigger_schema(trigger_plan.trigger_type)
    assert schema is not None
    trigger_plan.object_id = schema.object_id
    for prop in schema.properties:
        value = _property_value(trigger_plan, prop.name)
        if value is None and prop.default is not None:
            _set_property_value(trigger_plan, prop.name, prop.default)
        value = _property_value(trigger_plan, prop.name)
        if value is None:
            continue
        if prop.min_value is not None or prop.max_value is not None:
            low = prop.min_value if prop.min_value is not None else float(value)
            high = prop.max_value if prop.max_value is not None else float(value)
            _set_property_value(trigger_plan, prop.name, max(low, min(high, float(value))))
    return trigger_plan


def encode_trigger_properties_safe(
    trigger_plan: TriggerPlan,
    mode: TriggerMode | str | bool | None = TriggerMode.SAFE,
) -> dict[str, Any]:
    trigger_plan = apply_trigger_property_defaults(trigger_plan, mode)  # type: ignore[assignment]
    if trigger_plan is None:
        return {}
    issues = validate_trigger_properties(trigger_plan, mode)
    if issues:
        return {}
    schema = get_trigger_schema(trigger_plan.trigger_type)
    assert schema is not None
    encoded: dict[str, Any] = {"1": schema.object_id, "2": trigger_plan.x, "3": trigger_plan.y}
    for prop in schema.properties:
        if prop.save_key is None:
            continue
        if normalize_trigger_mode(mode) == TriggerMode.SAFE and not prop.safe_mode_allowed:
            continue
        value = _property_value(trigger_plan, prop.name)
        if value is None:
            continue
        if isinstance(value, bool):
            value = 1 if value else 0
        encoded[prop.save_key] = value
    return encoded


def _property_value(trigger_plan: TriggerPlan, name: str) -> Any:
    if hasattr(trigger_plan, name):
        return getattr(trigger_plan, name)
    return getattr(trigger_plan, "properties", {}).get(name)


def _set_property_value(trigger_plan: TriggerPlan, name: str, value: Any) -> None:
    if hasattr(trigger_plan, name):
        setattr(trigger_plan, name, value)
    elif hasattr(trigger_plan, "properties"):
        trigger_plan.properties[name] = value
