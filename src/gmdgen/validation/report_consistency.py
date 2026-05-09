# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ReportConsistencyError(ValueError):
    pass


@dataclass(slots=True)
class ReportConsistencyResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def validate_generation_report_consistency(
    report: dict[str, Any],
    *,
    hard: bool = False,
) -> ReportConsistencyResult:
    errors: list[str] = []
    warnings: list[str] = []

    raw_objects = _maybe_int(report, "raw_objects")
    candidate_objects = _maybe_int(report, "candidate_objects", "candidate_ir_objects")
    parsed_candidate_objects = _maybe_int(report, "parsed_candidate_objects", "parsed_objects")
    final_objects = _maybe_int(report, "final_objects", "final_object_count")
    serialized_objects = _maybe_int(report, "serialized_objects")

    is_fallback = bool(report.get("planner_fallback_used"))

    if _is_ollama_planner_report(report) and not is_fallback and raw_objects == 0 and _positive(candidate_objects, final_objects):
        errors.append("raw_objects_zero_but_candidate_or_final_objects_exist")
    if is_fallback and raw_objects == 0 and _positive(candidate_objects, final_objects):
        pass  # Fallback generates candidate/final objects deterministically, so raw_objects=0 is valid.

    if candidate_objects is not None and parsed_candidate_objects is not None and candidate_objects != parsed_candidate_objects:
        errors.append("candidate_objects_do_not_match_parsed_candidate_objects")
    
    fallback_generated_objects = _maybe_int(report, "fallback_generated_objects")
    
    if is_fallback:
        if fallback_generated_objects is not None and serialized_objects is not None and fallback_generated_objects != serialized_objects:
            errors.append("fallback_generated_objects_do_not_match_serialized_objects")
    else:
        if final_objects is not None and serialized_objects is not None and final_objects != serialized_objects:
            errors.append("final_objects_do_not_match_serialized_objects")
    
    if final_objects == 0 and not (is_fallback and fallback_generated_objects and fallback_generated_objects > 0):
        errors.append("final_objects_zero")

    if _metric(report, "missing_target_group", "unresolved_missing_target_group_count") > 0:
        errors.append("missing_target_group_positive")
    if _metric(report, "missing_color_channel") > 0:
        errors.append("missing_color_channel_positive")
    if _metric(report, "invalid_trigger_target", "orphan_trigger_count", "invalid_group_count") > 0:
        errors.append("invalid_trigger_target_positive")

    if bool(report.get("repair_applied")) and report.get("repair_metrics_updated") is False:
        errors.append("repair_metrics_not_updated_after_repair")
    if bool(report.get("low_quality_draft_saved")) and bool(report.get("final_success")):
        errors.append("low_quality_draft_marked_final_success")
    if bool(report.get("planner_fallback_used")) and bool(report.get("final_success")):
        errors.append("planner_fallback_marked_final_success")
    if not bool(report.get("ai_used")) and bool(report.get("final_success")) and not bool(report.get("local_fallback_used")):
        errors.append("ai_not_used_but_marked_final_success")
    if str(report.get("ai_fallback_reason", "")) == "ollama_forbidden_field":
        if not report.get("forbidden_fields"):
            errors.append("forbidden_fields_empty_despite_reason")
    if isinstance(report.get("candidate_reports"), list) and len(report["candidate_reports"]) == 0:
        if report.get("selected_candidate_id") is not None:
            errors.append("selected_candidate_id_not_null_without_candidates")

    # Timing consistency
    metrics = report.get("metrics", {})
    ai_calls_used = int(report.get("ai_calls_used", 0) or 0)
    ai_planning_seconds = float(metrics.get("ai_planning_seconds", 0.0) or 0.0)
    if ai_calls_used == 0 and ai_planning_seconds > 0.0:
        errors.append("ai_planning_seconds_without_ai_calls")

    # valid=true validation
    if bool(report.get("valid")) and not bool(report.get("final_success")) and report.get("planner_status") != "fallback":
         pass # Actually the requirement is valid=true SHOULD only mean serialized draft validation. We can check if it's distinct.

    if report.get("candidate_score_defined") is False or report.get("final_score_defined") is False:
        errors.append("candidate_or_final_score_definition_missing")

    for field_name in (
        "planner_status",
        "planner_fallback_used",
        "candidate_ir_objects",
        "serialized_objects",
        "final_objects",
        "syntax_validation",
        "semantic_validation",
        "playability_validation",
        "repair_applied",
        "repair_loss",
        "quality_gate_passed",
        "low_quality_draft_saved",
        "final_success",
    ):
        if field_name not in report:
            warnings.append(f"missing_report_contract_field:{field_name}")

    result = ReportConsistencyResult(passed=not errors, errors=errors, warnings=warnings)
    if hard and errors:
        raise ReportConsistencyError("; ".join(errors))
    return result


def contract_fields_from_generation_result(result: dict[str, Any]) -> dict[str, Any]:
    validation_report = result.get("validation_report", {})
    if not isinstance(validation_report, dict):
        validation_report = {}
    merged = {**validation_report, **result}
    plan_counts = validation_report.get("plan_count_report", {})
    if isinstance(plan_counts, dict):
        merged.setdefault("candidate_ir_objects", plan_counts.get("parsed_objects"))
        merged.setdefault("serialized_objects", plan_counts.get("final_encoded_objects"))

    is_fallback = bool(merged.get("planner_fallback_used"))
    if is_fallback:
        merged.setdefault("fallback_generated_objects", merged.get("final_object_count", merged.get("num_objects", 0)))
        merged.setdefault("final_objects", 0)  # AI output was 0
        merged.setdefault("raw_ai_object_count", 0)
    else:
        merged.setdefault("final_objects", merged.get("final_object_count", merged.get("num_objects", 0)))
        merged.setdefault("fallback_generated_objects", 0)

    merged.setdefault("quality_gate_passed", bool(merged.get("quality_gate_report", {}).get("passed", merged.get("quality_gate_passed", False))))
    merged.setdefault("low_quality_draft_saved", False)
    
    is_success = bool(merged.get("quality_gate_passed")) and not bool(merged.get("planner_fallback_used")) and bool(merged.get("ai_used", True))
    merged.setdefault("final_success", is_success)
    
    if not is_success:
        if bool(merged.get("planner_fallback_used")):
            merged.setdefault("final_success_reason", "planner_fallback_used")
        elif not bool(merged.get("quality_gate_passed")):
            merged.setdefault("final_success_reason", "quality_gate_failed")
        elif not bool(merged.get("ai_used", True)):
            merged.setdefault("final_success_reason", "ai_not_used")

    return merged


def _is_ollama_planner_report(report: dict[str, Any]) -> bool:
    status = str(report.get("planner_status", ""))
    provider = str(report.get("ai_provider", ""))
    return status in {"ollama_used", "valid"} or provider == "ollama"


def _positive(*values: int | None) -> bool:
    return any(value is not None and value > 0 for value in values)


def _maybe_int(report: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key in report and report[key] is not None:
            try:
                return int(report[key])
            except (TypeError, ValueError):
                return None
    return None


def _metric(report: dict[str, Any], *keys: str) -> int:
    metrics = report.get("metrics", {})
    for key in keys:
        value = report.get(key)
        if value is None and isinstance(metrics, dict):
            value = metrics.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return 0
