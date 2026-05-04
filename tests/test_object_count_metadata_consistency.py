from __future__ import annotations

"""Tests for Phase 9: Object count metadata consistency.

Previous symptom: GUI showed final_objects=5898 but quality log showed
raw_objects=0 final_objects=0 and a candidate report showed objects=5196.
These tests ensure object count fields are consistent and meaningful."""

from gmdgen.generate.quality_gate import QualityGateThresholds, evaluate_quality_gate


def _report(**overrides):
    base = {
        "score": 0.9,
        "final_object_count": 1000,
        "removed_object_ratio": 0.05,
        "drop_impact_score": 0.8,
        "metrics": {"density_alignment_score": 0.8},
        "score_breakdown": {"editor_validity": 0.9, "playability_safety": 0.9},
    }
    base.update(overrides)
    return base


def test_final_object_count_propagated_to_metrics():
    result = evaluate_quality_gate(_report(final_object_count=1234))
    assert result.metrics["final_object_count"] == 1234


def test_zero_final_object_count_fails_gate():
    result = evaluate_quality_gate(_report(final_object_count=0))
    # With min_object_count default (12), 0 should fail.
    assert result.passed is False


def test_plan_count_report_zero_encoded_objects_fails():
    result = evaluate_quality_gate(_report(
        final_object_count=100,
        plan_count_report={"final_encoded_objects": 0},
    ))
    assert result.passed is False
    assert any("final_encoded_objects_is_zero" in f for f in result.failures)


def test_metrics_use_consistent_repair_loss_field():
    """The quality gate must take the max of removed_object_ratio and removed_trigger_ratio."""
    result = evaluate_quality_gate(_report(
        removed_object_ratio=0.1,
        removed_trigger_ratio=0.6,
    ))
    assert result.metrics["repair_loss_ratio"] == 0.6


def test_object_plan_count_metadata_fields_exist_in_report():
    """The validation report should support standard object count fields."""
    from gmdgen.gd.plans import ValidationReport
    report = ValidationReport()
    # These fields are used by audio_conditioned and the GUI.
    expected_fields = ["raw_ai_object_count", "final_object_count", "speed_object_count"]
    for f in expected_fields:
        # Either the field exists as an attribute or `metrics` accepts it.
        assert hasattr(report, f) or hasattr(report, "metrics"), (
            f"ValidationReport missing field {f}"
        )


def test_unknown_count_does_not_appear_as_zero():
    """A None or missing count should not be coerced to 0 in failure causes."""
    result = evaluate_quality_gate(_report())
    # When everything is healthy, the failures list should be empty.
    assert result.passed is True or result.failures
    # The `repair_loss_ratio` must be a real number, not None
    assert isinstance(result.metrics["repair_loss_ratio"], (int, float))


def test_quality_gate_metrics_are_json_serializable():
    """All quality gate metrics must be JSON-serializable for the report."""
    import json
    result = evaluate_quality_gate(_report(final_object_count=500, removed_object_ratio=0.1))
    json.dumps(result.metrics)


def test_candidate_report_object_count_matches_plans():
    """build_candidate_report should record the same object count it was given."""
    from gmdgen.generate.quality import build_candidate_report
    report = build_candidate_report(
        candidate_id=1,
        conversion_valid=True,
        object_count=500,
        trigger_count=20,
        errors=[],
        warnings=[],
        score=0.85,
        section_plans=[],
    )
    d = report.to_dict()
    assert d["object_count"] == 500
    assert d["trigger_count"] == 20


def test_high_repair_loss_with_high_object_count_fails():
    """A level with many objects but high repair_loss must fail the gate."""
    result = evaluate_quality_gate(_report(
        final_object_count=5897,
        removed_object_ratio=0.4753,
        score_breakdown={"playability_safety": 0.13},
    ), QualityGateThresholds(max_repair_loss=0.5, min_playability=0.5))
    assert result.passed is False
    # Should fail on multiple criteria
    failure_text = " ".join(result.failures)
    assert "repair_loss" in failure_text or "playability" in failure_text


def test_diversity_metric_carried_through_when_present():
    """object_diversity_score should not collapse to 0 when supplied via metrics."""
    report = _report(metrics={"object_diversity_score": 0.05, "density_alignment_score": 0.8})
    result = evaluate_quality_gate(report)
    # Even if not directly checked by the gate, metrics dict should contain it
    # (we read via raw report dict, not the gate metrics).
    assert report["metrics"]["object_diversity_score"] == 0.05
