from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from gmdgen.features.tokenizer import extract_object_id, extract_object_number
from gmdgen.gd.plans import ObjectPlan, SectionPlan, TriggerPlan


@dataclass(slots=True)
class PlanSnapshot:
    stage: str
    section_count: int = 0
    object_count: int = 0
    trigger_count: int = 0
    role_distribution: dict[str, int] = field(default_factory=dict)
    object_id_distribution: dict[str, int] = field(default_factory=dict)
    trigger_type_distribution: dict[str, int] = field(default_factory=dict)
    density_by_section: dict[str, float] = field(default_factory=dict)
    average_objects_per_second: float = 0.0
    average_triggers_per_section: float = 0.0
    beat_aligned_event_count: int = 0
    onset_aligned_trigger_count: int = 0
    removed_object_count: int = 0
    removed_trigger_count: int = 0
    warnings: list[str] = field(default_factory=list)
    fatal_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlanDiff:
    from_stage: str
    to_stage: str
    removed_objects: int = 0
    removed_triggers: int = 0
    changed_roles: int = 0
    changed_trigger_properties: int = 0
    pruned_properties: int = 0
    normalized_easings: int = 0
    normalized_roles: int = 0
    density_change: float = 0.0
    style_score_change: float = 0.0
    sync_score_change: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RepairQualityReport:
    removed_due_to_unsupported_role: int = 0
    removed_due_to_unsupported_trigger: int = 0
    removed_due_to_missing_target_group: int = 0
    removed_due_to_object_budget: int = 0
    removed_due_to_overcrowding: int = 0
    removed_due_to_playability: int = 0
    removed_due_to_editor_safety: int = 0
    pruned_irrelevant_trigger_properties: int = 0
    clamped_duration_count: int = 0
    normalized_easing_count: int = 0
    normalized_object_role_count: int = 0
    density_smoothing_removed_count: int = 0
    safe_mode_pruned_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CandidateReport:
    candidate_id: int
    score: float = 0.0
    object_count: int = 0
    trigger_count: int = 0
    repair_loss_ratio: float = 0.0
    beat_sync: float = 0.0
    energy_density: float = 0.0
    drop_impact: float = 0.0
    editor_safety: float = 1.0
    selected: bool = False
    reject_reason: str = ""
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def snapshot_from_plans(
    stage: str,
    object_plans: Iterable[ObjectPlan | dict[str, Any]],
    trigger_plans: Iterable[TriggerPlan | dict[str, Any]],
    section_plans: list[SectionPlan] | None = None,
    *,
    warnings: list[str] | None = None,
    fatal_errors: list[str] | None = None,
) -> PlanSnapshot:
    objects = list(object_plans or [])
    triggers = list(trigger_plans or [])
    sections = section_plans or []
    roles: Counter[str] = Counter()
    object_ids: Counter[str] = Counter()
    trigger_types: Counter[str] = Counter()
    beat_aligned = 0
    onset_aligned = 0
    x_values: list[float] = []
    for plan in objects:
        role = _value(plan, "role", "unknown")
        object_id = _value(plan, "object_id", "unknown")
        roles[str(role)] += 1
        object_ids[str(object_id)] += 1
        if _value(plan, "beat_aligned_time", None) is not None:
            beat_aligned += 1
        x = _float_or_none(_value(plan, "x", None))
        if x is not None:
            x_values.append(x)
    for trigger in triggers:
        trigger_types[str(_value(trigger, "trigger_type", "unknown"))] += 1
        if _value(trigger, "beat_aligned_time", None) is not None:
            onset_aligned += 1
        x = _float_or_none(_value(trigger, "x", None))
        if x is not None:
            x_values.append(x)
    duration = max((section.end_time for section in sections), default=0.0) - min(
        (section.start_time for section in sections),
        default=0.0,
    )
    return PlanSnapshot(
        stage=stage,
        section_count=len(sections),
        object_count=len(objects),
        trigger_count=len(triggers),
        role_distribution=dict(roles),
        object_id_distribution=dict(object_ids),
        trigger_type_distribution=dict(trigger_types),
        density_by_section=compute_actual_density_by_section(objects, triggers, sections),
        average_objects_per_second=(len(objects) + len(triggers)) / duration if duration > 0 else 0.0,
        average_triggers_per_section=len(triggers) / len(sections) if sections else 0.0,
        beat_aligned_event_count=beat_aligned,
        onset_aligned_trigger_count=onset_aligned,
        warnings=list(warnings or []),
        fatal_errors=list(fatal_errors or []),
    )


def snapshot_from_level_objects(
    stage: str,
    level_objects: Iterable[str],
    section_plans: list[SectionPlan] | None = None,
    *,
    warnings: list[str] | None = None,
    fatal_errors: list[str] | None = None,
) -> PlanSnapshot:
    objects = list(level_objects or [])
    object_ids: Counter[str] = Counter()
    trigger_types: Counter[str] = Counter()
    x_values: list[float] = []
    trigger_ids = {"29", "33", "899", "901", "1006", "1007", "1268", "1347", "1520", "1611", "1815", "1817"}
    for obj in objects:
        object_id = extract_object_id(obj) or "unknown"
        object_ids[object_id] += 1
        if object_id in trigger_ids:
            trigger_types[object_id] += 1
        x = extract_object_number(obj, "2")
        if x is not None:
            x_values.append(float(x))
    section_count = len(section_plans or [])
    return PlanSnapshot(
        stage=stage,
        section_count=section_count,
        object_count=len(objects) - sum(trigger_types.values()),
        trigger_count=sum(trigger_types.values()),
        object_id_distribution=dict(object_ids),
        trigger_type_distribution=dict(trigger_types),
        density_by_section=compute_actual_density_by_section_from_objects(objects, section_plans or []),
        average_triggers_per_section=sum(trigger_types.values()) / section_count if section_count else 0.0,
        warnings=list(warnings or []),
        fatal_errors=list(fatal_errors or []),
    )


def diff_snapshots(previous: PlanSnapshot, current: PlanSnapshot) -> PlanDiff:
    return PlanDiff(
        from_stage=previous.stage,
        to_stage=current.stage,
        removed_objects=max(0, previous.object_count - current.object_count),
        removed_triggers=max(0, previous.trigger_count - current.trigger_count),
        density_change=sum(current.density_by_section.values()) - sum(previous.density_by_section.values()),
    )


def build_repair_quality_report(
    *,
    ai_normalization: Any | None = None,
    plan_validation_warnings: list[str] | None = None,
    string_repair_report: Any | None = None,
    editor_safety_fatal_count: int = 0,
) -> RepairQualityReport:
    warnings = plan_validation_warnings or []
    report = RepairQualityReport(
        pruned_irrelevant_trigger_properties=int(getattr(ai_normalization, "pruned_trigger_property_count", 0)),
        normalized_easing_count=int(getattr(ai_normalization, "normalized_easing_count", 0)),
        normalized_object_role_count=int(getattr(ai_normalization, "normalized_object_role_count", 0)),
        removed_due_to_editor_safety=editor_safety_fatal_count,
    )
    for warning in warnings:
        if "unsupported_role" in warning:
            report.removed_due_to_unsupported_role += 1
        if "removed_unsupported_trigger" in warning or "unsupported_trigger" in warning:
            report.removed_due_to_unsupported_trigger += 1
        if (
            "removed_orphan" in warning
            or "unresolved_missing_target_group" in warning
            or "trigger_missing_target_group" in warning
        ):
            report.removed_due_to_missing_target_group += 1
        if "object_budget" in warning:
            report.removed_due_to_object_budget += 1
        if "crowded" in warning:
            report.removed_due_to_overcrowding += 1
    if string_repair_report is not None:
        report.removed_due_to_object_budget += int(getattr(string_repair_report, "budget_pruned", 0))
        report.removed_due_to_overcrowding += int(getattr(string_repair_report, "density_spread", 0))
        report.removed_due_to_missing_target_group += int(getattr(string_repair_report, "orphan_trigger_removed", 0))
        report.removed_due_to_unsupported_trigger += int(getattr(string_repair_report, "trigger_schema_removed", 0))
        report.clamped_duration_count += int(getattr(string_repair_report, "trigger_schema_repaired", 0))
        report.safe_mode_pruned_count += int(getattr(string_repair_report, "unsafe_trigger_removed", 0))
        report.removed_due_to_playability += int(getattr(string_repair_report, "playability_pruned", 0))
    return report


def summarize_quality_loss(
    report: RepairQualityReport,
    *,
    raw_object_count: int,
    final_object_count: int,
    raw_trigger_count: int,
    final_trigger_count: int,
    drop_impact_score: float = 1.0,
) -> list[str]:
    reasons: list[str] = []
    total_removed = max(0, raw_object_count + raw_trigger_count - final_object_count - final_trigger_count)
    original_total = max(1, raw_object_count + raw_trigger_count)
    removed_ratio = total_removed / original_total
    if report.pruned_irrelevant_trigger_properties:
        reasons.append(
            f"Trigger properties were pruned {report.pruned_irrelevant_trigger_properties} times because they were not allowed for that trigger type."
        )
    if report.removed_due_to_object_budget or removed_ratio >= 0.25:
        reasons.append(
            f"Final map is sparse because pruning removed {round(removed_ratio * 100)}% of planned objects/triggers."
        )
    if report.removed_due_to_missing_target_group:
        reasons.append(
            f"{report.removed_due_to_missing_target_group} triggers were removed because target groups were missing."
        )
    if report.removed_due_to_unsupported_trigger:
        reasons.append(
            f"{report.removed_due_to_unsupported_trigger} unsupported triggers were removed by safe-mode validation."
        )
    if drop_impact_score < 0.35:
        reasons.append("Drop section lost impact because density or trigger intensity was too low.")
    if not reasons:
        reasons.append("No major repair loss detected.")
    return reasons[:5]


def compute_actual_density_by_section(
    object_plans: Iterable[ObjectPlan | dict[str, Any]],
    trigger_plans: Iterable[TriggerPlan | dict[str, Any]],
    section_plans: list[SectionPlan],
) -> dict[str, float]:
    result = {str(idx): 0.0 for idx in range(len(section_plans))}
    if not section_plans:
        return result
    counts = {idx: 0 for idx in range(len(section_plans))}
    for plan in list(object_plans or []) + list(trigger_plans or []):
        x = _float_or_none(_value(plan, "x", None))
        if x is None:
            continue
        idx = _section_index_for_x(x, section_plans)
        counts[idx] += 1
    for idx, count in counts.items():
        section = section_plans[idx]
        width = max(1.0, section.end_x - section.start_x)
        result[str(idx)] = round(count / width * 1000.0, 4)
    return result


def compute_actual_density_by_section_from_objects(
    level_objects: Iterable[str],
    section_plans: list[SectionPlan],
) -> dict[str, float]:
    result = {str(idx): 0.0 for idx in range(len(section_plans))}
    if not section_plans:
        return result
    counts = {idx: 0 for idx in range(len(section_plans))}
    for obj in level_objects:
        x = extract_object_number(obj, "2")
        if x is None:
            continue
        idx = _section_index_for_x(float(x), section_plans)
        counts[idx] += 1
    for idx, count in counts.items():
        section = section_plans[idx]
        width = max(1.0, section.end_x - section.start_x)
        result[str(idx)] = round(count / width * 1000.0, 4)
    return result


def compute_density_target_by_section(section_plans: list[SectionPlan]) -> dict[str, float]:
    return {str(idx): round(section.density_target, 4) for idx, section in enumerate(section_plans)}


def compute_density_target_error(target: dict[str, float], actual: dict[str, float]) -> float:
    if not target:
        return 0.0
    max_actual = max(actual.values(), default=1.0) or 1.0
    errors = []
    for key, target_value in target.items():
        normalized_actual = actual.get(key, 0.0) / max_actual
        errors.append(abs(float(target_value) - normalized_actual))
    return round(sum(errors) / len(errors), 4)


def compute_drop_impact_score(
    section_plans: list[SectionPlan],
    actual_density: dict[str, float],
    trigger_counts: dict[str, int] | None = None,
) -> float:
    drop_indices = [idx for idx, section in enumerate(section_plans) if section.section_type == "drop"]
    if not drop_indices:
        return 1.0
    all_density = list(actual_density.values())
    baseline = sum(all_density) / len(all_density) if all_density else 0.0
    if baseline <= 0:
        return 0.0
    values = []
    for idx in drop_indices:
        density_ratio = actual_density.get(str(idx), 0.0) / baseline
        trigger_bonus = min(0.35, (trigger_counts or {}).get(str(idx), 0) * 0.05)
        values.append(min(1.0, density_ratio / 1.35 + trigger_bonus))
    return round(sum(values) / len(values), 4)


def compute_buildup_progression_score(
    section_plans: list[SectionPlan],
    actual_density: dict[str, float],
) -> float:
    buildup = [
        actual_density.get(str(idx), 0.0)
        for idx, section in enumerate(section_plans)
        if section.section_type == "buildup"
    ]
    if len(buildup) < 2:
        return 1.0
    nondecreasing = sum(1 for left, right in zip(buildup, buildup[1:]) if right >= left)
    return round(nondecreasing / (len(buildup) - 1), 4)


def build_candidate_report(
    *,
    candidate_id: int,
    conversion_valid: bool,
    object_count: int,
    trigger_count: int,
    errors: list[str],
    warnings: list[str],
    section_plans: list[SectionPlan],
    score: float | None = None,
) -> CandidateReport:
    snapshot = snapshot_from_plans(
        "candidate",
        [],
        [],
        section_plans,
        warnings=warnings,
        fatal_errors=errors,
    )
    density = snapshot.density_by_section
    drop_impact = compute_drop_impact_score(section_plans, density)
    computed_score = score if score is not None else candidate_quality_score(
        object_count=object_count,
        trigger_count=trigger_count,
        error_count=len(errors),
        warning_count=len(warnings),
        drop_impact=drop_impact,
    )
    return CandidateReport(
        candidate_id=candidate_id,
        score=round(computed_score, 4),
        object_count=object_count,
        trigger_count=trigger_count,
        drop_impact=drop_impact,
        editor_safety=0.0 if errors else 1.0,
        reject_reason="; ".join(errors[:3]) if errors else "",
        warnings=list(warnings[:12]),
        errors=list(errors[:12]),
    )


def candidate_quality_score(
    *,
    object_count: int,
    trigger_count: int,
    error_count: int,
    warning_count: int,
    drop_impact: float,
) -> float:
    density_score = min(1.0, object_count / 24.0)
    trigger_score = min(1.0, trigger_count / 6.0)
    penalty = min(0.8, error_count * 0.35 + warning_count * 0.015)
    return max(0.0, 0.42 * density_score + 0.18 * trigger_score + 0.25 * drop_impact + 0.15 - penalty)


def build_quality_feedback_prompt(candidate_report: CandidateReport, quality_loss_reasons: list[str] | None = None) -> str:
    reasons = list(quality_loss_reasons or [])
    if candidate_report.reject_reason:
        reasons.append(candidate_report.reject_reason)
    if candidate_report.object_count < 8:
        reasons.append("too_sparse: add more valid beat-aligned gameplay and structure objects")
    if candidate_report.drop_impact < 0.35:
        reasons.append("drop_too_weak: make drop sections denser with valid pulse/color accents")
    if candidate_report.warnings:
        reasons.append("warnings: " + "; ".join(candidate_report.warnings[:3]))
    message = "Previous candidate quality feedback: " + " | ".join(reasons[:6])
    return message[:1200]


def _value(plan: Any, key: str, default: Any = None) -> Any:
    if isinstance(plan, dict):
        return plan.get(key, default)
    return getattr(plan, key, default)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _section_index_for_x(x: float, section_plans: list[SectionPlan]) -> int:
    for idx, section in enumerate(section_plans):
        if section.start_x <= x <= section.end_x:
            return idx
    if not section_plans:
        return 0
    return min(range(len(section_plans)), key=lambda idx: abs(section_plans[idx].start_x - x))
