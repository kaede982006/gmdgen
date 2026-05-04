from __future__ import annotations

from gmdgen.generate.quality_gate import QualityGateThresholds, QualityGateResult, evaluate_quality_gate


def _good_report(**overrides) -> dict:
    base = {
        "score": 0.9,
        "final_object_count": 40,
        "removed_object_ratio": 0.05,
        "drop_impact_score": 0.8,
        "metrics": {"density_alignment_score": 0.8},
        "score_breakdown": {"editor_validity": 0.9, "playability_safety": 0.9},
    }
    base.update(overrides)
    return base


def test_quality_gate_result_has_required_fields():
    result = evaluate_quality_gate(_good_report())
    d = result.to_dict()
    required_keys = [
        "passed", "failures", "warnings", "metrics",
        "primary_causes", "recommended_actions",
        "can_retry", "can_regenerate_weak_sections",
    ]
    for key in required_keys:
        assert key in d, f"Missing required field '{key}' in QualityGateResult"


def test_metrics_includes_core_scores():
    result = evaluate_quality_gate(_good_report(removed_object_ratio=0.3))
    metrics = result.metrics
    assert "final_score" in metrics
    assert "final_object_count" in metrics
    assert "repair_loss_ratio" in metrics
    assert "drop_impact_score" in metrics
    assert "density_alignment_score" in metrics
    assert "editor_safety_score" in metrics
    assert "playability_safety_score" in metrics


def test_failure_report_includes_stopped_reason():
    report = _good_report(stopped_reason="max_retries_exceeded")
    result = evaluate_quality_gate(report)
    assert any("max_retries_exceeded" in c for c in result.primary_causes)


def test_weak_sections_populated_when_density_low():
    report = _good_report(
        actual_density_by_section={"verse_1": 0.01},
        density_target_by_section={"verse_1": 0.5},
    )
    result = evaluate_quality_gate(report)
    assert "verse_1" in result.weak_sections


def test_retry_prompt_summary_populated_on_failure():
    report = _good_report(removed_object_ratio=0.8)
    result = evaluate_quality_gate(report)
    assert result.retry_prompt_summary, "Expected a retry_prompt_summary on failure"
    assert len(result.retry_prompt_summary) <= 1200


def test_can_retry_is_true_when_failures_exist():
    report = _good_report(score=0.1)
    result = evaluate_quality_gate(report)
    if result.failures:
        assert result.can_retry is True


def test_can_regenerate_weak_sections_when_weak_sections_exist():
    report = _good_report(
        actual_density_by_section={"intro": 0.0},
        density_target_by_section={"intro": 0.6},
    )
    result = evaluate_quality_gate(report)
    if result.weak_sections:
        assert result.can_regenerate_weak_sections is True


def test_empty_important_sections_reported():
    report = _good_report(empty_important_sections=["drop_1"])
    result = evaluate_quality_gate(report, QualityGateThresholds(reject_empty_important_sections=True, min_object_count=1))
    assert result.passed is False
    assert any("empty_important_sections" in f for f in result.failures)


def test_geode_fatal_issue_reported():
    report = _good_report(geode_fatal_issue_count=1)
    result = evaluate_quality_gate(report, QualityGateThresholds(require_no_fatal_geode_issue=True, min_object_count=1))
    assert result.passed is False
    assert any("geode" in f for f in result.failures)


def test_quality_gate_result_is_dataclass():
    """QualityGateResult must be instantiable directly for test fixture construction."""
    r = QualityGateResult(passed=True, failures=[], warnings=["test warning"])
    assert r.passed is True
    assert r.warnings == ["test warning"]
    d = r.to_dict()
    assert d["warnings"] == ["test warning"]


def test_report_with_repair_loss_breakdown_captured():
    breakdown = {"x_mono_fixes": 300, "orphan_triggers": 50}
    report = _good_report(
        removed_object_ratio=0.7,
        repair_loss_breakdown=breakdown,
    )
    result = evaluate_quality_gate(report)
    assert result.repair_loss_breakdown == breakdown


def test_plan_count_report_zero_objects_fails():
    report = _good_report(
        plan_count_report={"final_encoded_objects": 0},
        final_object_count=0,
    )
    result = evaluate_quality_gate(report)
    assert result.passed is False
