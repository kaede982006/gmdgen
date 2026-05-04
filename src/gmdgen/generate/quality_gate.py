from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class QualityGateThresholds:
    min_score: float = 0.45
    min_object_count: int = 12
    max_repair_loss: float = 0.5
    min_drop_impact: float = 0.35
    min_density_alignment: float = 0.35
    min_editor_safety: float = 0.75
    min_playability: float = 0.65
    require_no_fatal_geode_issue: bool = True
    reject_empty_important_sections: bool = True


@dataclass(slots=True)
class QualityGateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    repair_loss_breakdown: dict[str, Any] = field(default_factory=dict)
    playability_breakdown: dict[str, Any] = field(default_factory=dict)
    primary_causes: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    trigger_pruning_count: int = 0
    missing_target_group_count: int = 0
    weak_sections: list[str] = field(default_factory=list)
    retry_prompt_summary: str = ""
    can_retry: bool = False
    can_regenerate_weak_sections: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_quality_gate(report: dict[str, Any], thresholds: QualityGateThresholds | None = None) -> QualityGateResult:
    thresholds = thresholds or QualityGateThresholds()
    failures: list[str] = []
    metrics = _metrics(report)
    
    if metrics["final_score"] < thresholds.min_score:
        failures.append(f"final_score_below_threshold: {metrics['final_score']:.4f} < {thresholds.min_score}")
    
    if metrics["final_object_count"] < thresholds.min_object_count:
        failures.append(f"final_object_count_below_threshold: {metrics['final_object_count']} < {thresholds.min_object_count}")
        
    if metrics["repair_loss_ratio"] > thresholds.max_repair_loss:
        failures.append(f"repair_loss_above_threshold: {metrics['repair_loss_ratio']:.4f} > {thresholds.max_repair_loss}")
        
    if metrics["drop_impact_score"] < thresholds.min_drop_impact:
        failures.append(f"drop_impact_below_threshold: {metrics['drop_impact_score']:.4f} < {thresholds.min_drop_impact}")
        
    if metrics["density_alignment_score"] < thresholds.min_density_alignment:
        failures.append(f"density_alignment_below_threshold: {metrics['density_alignment_score']:.4f} < {thresholds.min_density_alignment}")
        
    if metrics["editor_safety_score"] < thresholds.min_editor_safety:
        failures.append(f"editor_safety_below_threshold: {metrics['editor_safety_score']:.4f} < {thresholds.min_editor_safety}")
        
    if metrics["playability_safety_score"] < thresholds.min_playability:
        failures.append(f"playability_below_threshold: {metrics['playability_safety_score']:.4f} < {thresholds.min_playability}")
        
    if thresholds.require_no_fatal_geode_issue and report.get("geode_fatal_issue_count", 0):
        failures.append("fatal_geode_issue_present")
        
    if thresholds.reject_empty_important_sections:
        empty = [item for item in report.get("empty_important_sections", []) if item]
        if empty:
            failures.append(f"empty_important_sections: {empty}")
            
    plan_count_report = report.get("plan_count_report", {})
    if isinstance(plan_count_report, dict):
        final_encoded_objects = plan_count_report.get("final_encoded_objects")
        if final_encoded_objects == 0:
            failures.append("final_encoded_objects_is_zero")

    repair_loss_breakdown = report.get("repair_loss_breakdown", {})
    playability_breakdown = report.get("playability_breakdown", {})
    trigger_pruning_count = int(report.get("pruned_trigger_property_count", 0) or 0)
    repair_quality = report.get("repair_quality_report", {}) if isinstance(report.get("repair_quality_report", {}), dict) else {}
    missing_target_group_count = int(
        repair_quality.get(
            "removed_due_to_missing_target_group",
            metrics.get("unresolved_missing_target_group_count", 0),
        )
        or 0
    )
    weak_sections = _weak_sections(report)
    
    primary_causes = []
    recommended_actions = []
    
    # Add stopped reason to causes if it exists
    stopped_reason = report.get("stopped_reason") or report.get("validation_report", {}).get("stopped_reason")
    if stopped_reason:
        primary_causes.append(f"Generation stopped: {stopped_reason}")

    if metrics["repair_loss_ratio"] > thresholds.max_repair_loss:
        causes = [k for k, v in repair_loss_breakdown.items() if isinstance(v, (int, float)) and v > 0]
        primary_causes.extend([f"High repair loss: {c}" for c in causes[:3]] if causes else ["High repair loss from validation failures."])
        recommended_actions.extend(["Use fewer but valid triggers.", "Do not invent trigger properties.", "Keep irrelevant trigger fields null."])
        
    if metrics["playability_safety_score"] < thresholds.min_playability:
        # Check specific playability warnings in breakdown
        dense = playability_breakdown.get("excessive_input_density_count", 0)
        tight = playability_breakdown.get("tight_spacing_count", 0)
        if dense: primary_causes.append(f"Too many inputs in small window ({dense} warnings)")
        if tight: primary_causes.append(f"Objects are too close together ({tight} warnings)")
        
        primary_causes.extend(["Dense inputs or unsafe spacing."])
        recommended_actions.extend(["Increase spacing between gameplay events.", "Reduce tight orb/pad chains.", "Add safer transition margins after portals."])
    
    if metrics["final_object_count"] < thresholds.min_object_count:
        recommended_actions.append("Increase object_multiplier or decoration_density.")

    if trigger_pruning_count:
        primary_causes.append(f"Trigger pruning count is high: {trigger_pruning_count}")
        recommended_actions.append("Ask Ollama for trigger intents and null irrelevant trigger fields.")
        
    if missing_target_group_count:
        primary_causes.append(f"Missing target groups affected {missing_target_group_count} triggers.")
        recommended_actions.append("Regenerate weak sections with target_role and section_id hints.")
        
    if weak_sections:
        recommended_actions.append("Regenerate weak sections only before retrying the full level.")
        
    retry_prompt_summary = _retry_prompt_summary(
        failures,
        primary_causes,
        trigger_pruning_count=trigger_pruning_count,
        missing_target_group_count=missing_target_group_count,
        weak_sections=weak_sections,
    )

    return QualityGateResult(
        passed=not failures,
        failures=failures,
        metrics=metrics,
        repair_loss_breakdown=repair_loss_breakdown,
        playability_breakdown=playability_breakdown,
        primary_causes=primary_causes,
        recommended_actions=list(dict.fromkeys(recommended_actions)),
        trigger_pruning_count=trigger_pruning_count,
        missing_target_group_count=missing_target_group_count,
        weak_sections=weak_sections,
        retry_prompt_summary=retry_prompt_summary,
        can_retry=bool(failures),
        can_regenerate_weak_sections=bool(weak_sections),
    )


def _metrics(report: dict[str, Any]) -> dict[str, Any]:
    score_breakdown = report.get("score_breakdown", {}) if isinstance(report.get("score_breakdown", {}), dict) else {}
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics", {}), dict) else {}
    removed_object = float(report.get("removed_object_ratio", 0.0) or 0.0)
    removed_trigger = float(report.get("removed_trigger_ratio", 0.0) or 0.0)
    return {
        "final_score": float(report.get("score", score_breakdown.get("total", 0.0)) or 0.0),
        "final_object_count": int(report.get("final_object_count", 0) or 0),
        "repair_loss_ratio": max(removed_object, removed_trigger),
        "drop_impact_score": float(report.get("drop_impact_score", metrics.get("drop_impact_score", 0.0)) or 0.0),
        "density_alignment_score": float(metrics.get("density_alignment_score", score_breakdown.get("density_alignment", 0.0)) or 0.0),
        "editor_safety_score": float(score_breakdown.get("editor_validity", score_breakdown.get("editor_safety", 1.0)) or 1.0),
        "playability_safety_score": float(score_breakdown.get("playability_safety", 1.0) or 1.0),
        "unresolved_missing_target_group_count": int(metrics.get("unresolved_missing_target_group_count", 0) or 0),
    }


def _weak_sections(report: dict[str, Any]) -> list[str]:
    actual = report.get("actual_density_by_section", {})
    target = report.get("density_target_by_section", {})
    weak: list[str] = []
    if isinstance(actual, dict) and isinstance(target, dict):
        for section_id, target_value in target.items():
            try:
                if float(actual.get(section_id, 0.0)) < float(target_value) * 0.35:
                    weak.append(str(section_id))
            except Exception:
                continue
    empty = report.get("empty_important_sections", [])
    if isinstance(empty, list):
        weak.extend(str(item) for item in empty if item)
    return list(dict.fromkeys(weak))[:8]


def _retry_prompt_summary(
    failures: list[str],
    primary_causes: list[str],
    *,
    trigger_pruning_count: int,
    missing_target_group_count: int,
    weak_sections: list[str],
) -> str:
    parts: list[str] = []
    if failures:
        parts.append("failed_checks=" + ", ".join(failures[:4]))
    if trigger_pruning_count:
        parts.append(f"trigger_pruning_count={trigger_pruning_count}; prefer trigger intents and null irrelevant fields")
    if missing_target_group_count:
        parts.append(f"missing_target_group_count={missing_target_group_count}; include target_role/section_id or rely on materializer")
    if weak_sections:
        parts.append("weak_sections=" + ", ".join(weak_sections[:5]))
    if primary_causes:
        parts.append("primary_causes=" + " | ".join(primary_causes[:3]))
    return "; ".join(parts)[:1200]
