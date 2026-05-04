from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from gmdgen.gd.plans import TriggerMode, TriggerPlan
from gmdgen.gd.triggers import (
    apply_trigger_property_defaults,
    get_trigger_schema,
    reject_unsupported_trigger,
)


@dataclass(slots=True)
class TriggerIntent:
    trigger_kind: str
    purpose: str = "beat_accent"
    target_role: str = "section_group"
    intensity: float = 0.5
    duration_hint: float | None = None
    beat_aligned_time: float | None = None
    section_id: int | None = None


@dataclass(slots=True)
class TriggerMaterializationReport:
    materialized_count: int = 0
    pruned_irrelevant_property_count: int = 0
    defaulted_target_group_count: int = 0
    rejected_count: int = 0
    warnings: list[str] = field(default_factory=list)


def materialize_trigger_roles(
    trigger_plans: list[TriggerPlan],
    *,
    safe_mode: bool = True,
    seed: int = 42,
) -> list[TriggerPlan]:
    rng = random.Random(seed)
    report = TriggerMaterializationReport()
    materialized: list[TriggerPlan] = []
    for plan in trigger_plans:
        defaulted = materialize_trigger_plan(
            plan,
            safe_mode=safe_mode,
            seed=rng.randint(0, 1_000_000),
            report=report,
        )
        if defaulted is not None:
            materialized.append(defaulted)
    trigger_plans[:] = materialized
    return materialized


def materialize_trigger_plan(
    plan: TriggerPlan,
    *,
    safe_mode: bool = True,
    seed: int = 42,
    report: TriggerMaterializationReport | None = None,
) -> TriggerPlan | None:
    _rng = random.Random(seed)
    mode = TriggerMode.SAFE if safe_mode else TriggerMode.ADVANCED
    schema = get_trigger_schema(plan.trigger_type)
    if schema is None or reject_unsupported_trigger(plan, mode):
        if report:
            report.rejected_count += 1
        return None

    intent = trigger_intent_from_plan(plan)
    plan.object_id = schema.object_id
    allowed = {prop.name for prop in schema.properties}
    for name in sorted(list(plan.properties.keys())):
        if name not in allowed:
            plan.properties.pop(name, None)
            if report:
                report.pruned_irrelevant_property_count += 1
                report.warnings.append(f"materializer_pruned_irrelevant_property: {plan.trigger_type}.{name}")

    if plan.target_group is None and intent.target_role == "test_default_group":
        plan.target_group = 1
        if report:
            report.defaulted_target_group_count += 1

    if plan.duration <= 0 and intent.duration_hint is not None:
        plan.duration = max(0.02, min(2.0, intent.duration_hint))
    if plan.trigger_type == "move":
        plan.easing = _normalize_easing(plan.easing) or "linear"
        if plan.properties.get("move_x") is None and plan.properties.get("move_y") is None:
            plan.properties["move_y"] = round(8.0 + 24.0 * intent.intensity, 3)
    elif plan.trigger_type in {"pulse", "color"} and plan.properties.get("color_channel") is None:
        plan.properties["color_channel"] = 1
    elif plan.trigger_type == "alpha" and plan.properties.get("opacity") is None:
        plan.properties["opacity"] = round(max(0.15, min(1.0, 0.35 + intent.intensity * 0.55)), 3)

    defaulted = apply_trigger_property_defaults(plan, mode)
    if defaulted is not None and report:
        report.materialized_count += 1
    return defaulted


def materialize_trigger_intent(
    intent: TriggerIntent,
    *,
    x: float,
    y: float,
    target_group: int | None = None,
    safe_mode: bool = True,
) -> TriggerPlan | None:
    trigger_type = intent.trigger_kind.strip().lower()
    schema = get_trigger_schema(trigger_type)
    if schema is None:
        return None
    plan = TriggerPlan(
        trigger_type=trigger_type,
        object_id=schema.object_id,
        x=x,
        y=y,
        target_group=target_group,
        duration=intent.duration_hint or 0.0,
        beat_aligned_time=intent.beat_aligned_time,
        properties={},
    )
    return materialize_trigger_plan(plan, safe_mode=safe_mode)


def trigger_intent_from_plan(plan: TriggerPlan) -> TriggerIntent:
    properties = plan.properties or {}
    return TriggerIntent(
        trigger_kind=str(properties.pop("trigger_kind", None) or plan.trigger_type).strip().lower(),
        purpose=str(properties.pop("purpose", None) or "beat_accent").strip().lower(),
        target_role=str(properties.pop("target_role", None) or "section_group").strip().lower(),
        intensity=_clamp_float(properties.pop("intensity", 0.5), 0.0, 1.0, 0.5),
        duration_hint=_optional_float(properties.pop("duration_hint", None)),
        beat_aligned_time=plan.beat_aligned_time,
        section_id=_optional_int(properties.pop("section_id", None)),
    )


def _normalize_easing(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    key = raw.replace(" ", "").replace("_", "").replace("-", "").lower()
    return {
        "none": "linear",
        "linear": "linear",
        "easein": "ease_in",
        "easeout": "ease_out",
        "easeinout": "ease_in_out",
        "inout": "ease_in_out",
    }.get(key, "linear")


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
