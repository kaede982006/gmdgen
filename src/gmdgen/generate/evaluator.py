# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from gmdgen.gd.geode_bridge import GeodeBridge, NullGeodeBridge
from gmdgen.gd.plans import ObjectPlan, SectionPlan, TriggerPlan
from gmdgen.generate.quality import (
    compute_actual_density_by_section,
    compute_density_target_by_section,
    compute_density_target_error,
    compute_drop_impact_score,
)
from gmdgen.generate.role_mapping import object_diversity_for_plans


@dataclass(slots=True)
class GeodeQualityReport:
    available: bool = False
    version: str = ""
    parse_ok: bool = False
    round_trip_ok: bool = False
    import_safety_ok: bool = False
    time_x_avg_error: float = 0.0
    time_x_max_error: float = 0.0
    unsupported_object_count: int = 0
    unsupported_trigger_count: int = 0
    crash_risk_count: int = 0
    warnings: list[str] = field(default_factory=list)
    score_penalty: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SectionEvaluatorReport:
    section_id: int
    beat_sync: float = 0.0
    onset_sync: float = 0.0
    energy_density: float = 0.0
    section_contrast: float = 0.0
    motif_quality: float = 0.0
    object_diversity: float = 0.0
    trigger_usefulness: float = 0.0
    playability_safety: float = 1.0
    editor_safety: float = 1.0
    style_match: float = 0.0
    repair_loss_penalty: float = 0.0
    geode_penalty: float = 0.0
    total: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_section_candidate(
    *,
    section: SectionPlan,
    object_plans: list[ObjectPlan],
    trigger_plans: list[TriggerPlan],
    style_summary: dict[str, Any] | None = None,
    repair_loss_ratio: float = 0.0,
    geode_report: GeodeQualityReport | None = None,
) -> SectionEvaluatorReport:
    density = compute_actual_density_by_section(object_plans, trigger_plans, [section])
    target = compute_density_target_by_section([section])
    density_error = compute_density_target_error(target, density)
    density_alignment = max(0.0, 1.0 - density_error)
    object_diversity = object_diversity_for_plans(object_plans)
    beat_aligned = sum(1 for plan in object_plans if plan.beat_aligned_time is not None)
    onset_aligned = sum(1 for trigger in trigger_plans if trigger.beat_aligned_time is not None)
    beat_sync = min(1.0, beat_aligned / max(1, len(object_plans)))
    onset_sync = min(1.0, onset_aligned / max(1, len(trigger_plans))) if trigger_plans else 0.5
    trigger_usefulness = min(1.0, len(trigger_plans) / max(1.0, section.trigger_intensity * 6.0))
    style_match = _style_match_score(object_plans, style_summary or {})
    motif_quality = _motif_quality(object_plans, trigger_plans, section)
    section_contrast = 1.0 if section.section_type in {"drop", "buildup", "break"} else 0.65
    geode_penalty = float(geode_report.score_penalty if geode_report else 0.0)
    report = SectionEvaluatorReport(
        section_id=_section_id(section),
        beat_sync=round(beat_sync, 4),
        onset_sync=round(onset_sync, 4),
        energy_density=round(density_alignment, 4),
        section_contrast=round(section_contrast, 4),
        motif_quality=round(motif_quality, 4),
        object_diversity=round(object_diversity, 4),
        trigger_usefulness=round(trigger_usefulness, 4),
        style_match=round(style_match, 4),
        repair_loss_penalty=round(min(1.0, max(0.0, repair_loss_ratio)), 4),
        geode_penalty=round(min(1.0, max(0.0, geode_penalty)), 4),
    )
    if section.section_type == "drop" and (len(object_plans) < 4 or len(trigger_plans) < 1):
        report.warnings.append("drop_section_low_impact")
    if not object_plans:
        report.warnings.append("empty_section")
    report.total = round(
        max(
            0.0,
            0.14 * report.beat_sync
            + 0.10 * report.onset_sync
            + 0.16 * report.energy_density
            + 0.10 * report.section_contrast
            + 0.14 * report.motif_quality
            + 0.10 * report.object_diversity
            + 0.08 * report.trigger_usefulness
            + 0.10 * report.style_match
            + 0.08 * report.editor_safety
            - 0.18 * report.repair_loss_penalty
            - 0.16 * report.geode_penalty
            - 0.05 * len(report.warnings),
        ),
        4,
    )
    return report


def evaluate_geode_quality(
    geode_bridge: GeodeBridge | None,
    *,
    level_string: str = "",
    time_x_avg_error: float = 0.0,
    time_x_max_error: float = 0.0,
) -> GeodeQualityReport:
    bridge = geode_bridge or NullGeodeBridge()
    if not bridge.is_available():
        return GeodeQualityReport(warnings=["geode_unavailable"], score_penalty=0.0)
    report = GeodeQualityReport(
        available=True,
        version=bridge.get_version() or "",
        time_x_avg_error=float(time_x_avg_error),
        time_x_max_error=float(time_x_max_error),
    )
    if level_string:
        validation = bridge.validate_level_string(level_string)
        round_trip = bridge.round_trip_level_string(level_string)
        import_safety = bridge.import_safety_check(level_string)
        trigger_report = bridge.inspect_triggers(level_string)
        report.parse_ok = bool(validation.valid)
        report.round_trip_ok = bool(round_trip.round_trip_ok)
        report.import_safety_ok = bool(import_safety.safe)
        report.unsupported_trigger_count = int(trigger_report.unsupported_trigger_count)
        report.warnings.extend(validation.warnings)
        report.warnings.extend(round_trip.warnings)
        report.warnings.extend(import_safety.warnings)
        report.crash_risk_count = len(validation.fatal_errors) + len(import_safety.fatal_errors)
    report.score_penalty = min(
        1.0,
        0.05 * len(report.warnings)
        + 0.1 * report.unsupported_trigger_count
        + 0.25 * report.crash_risk_count
        + min(0.25, report.time_x_max_error),
    )
    return report


def combine_model_critic_and_deterministic_scores(
    deterministic_score: float,
    critic_score: float | None,
    *,
    critic_weight: float = 0.35,
) -> float:
    deterministic_score = _clamp01(deterministic_score)
    if critic_score is None:
        return deterministic_score
    critic_score = _clamp01(critic_score)
    weight = _clamp01(critic_weight)
    return round((1.0 - weight) * deterministic_score + weight * critic_score, 4)


def _style_match_score(object_plans: list[ObjectPlan], style_summary: dict[str, Any]) -> float:
    distribution = style_summary.get("object_id_distribution", {})
    if not isinstance(distribution, dict) or not distribution:
        return 0.55 if object_plans else 0.0
    hits = sum(1 for plan in object_plans if str(plan.object_id) in distribution)
    return min(1.0, hits / max(1, len(object_plans)) + 0.25)


def _motif_quality(object_plans: list[ObjectPlan], trigger_plans: list[TriggerPlan], section: SectionPlan) -> float:
    if not object_plans:
        return 0.0
    roles = {plan.role for plan in object_plans}
    base = min(1.0, len(roles) / 4.0)
    if section.section_type == "drop":
        base += 0.2 if trigger_plans else -0.2
    if section.section_type == "break":
        base += 0.15 if len(object_plans) <= 5 else -0.1
    return _clamp01(base)


def _section_id(section: SectionPlan) -> int:
    try:
        return int(getattr(section, "section_id"))  # type: ignore[arg-type]
    except Exception:
        return 0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
