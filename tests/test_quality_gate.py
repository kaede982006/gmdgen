from __future__ import annotations

from gmdgen.generate.quality_gate import QualityGateThresholds, evaluate_quality_gate


def test_quality_gate_rejects_too_sparse_level() -> None:
    result = evaluate_quality_gate({"score": 0.9, "final_object_count": 2}, QualityGateThresholds(min_object_count=10))

    assert result.passed is False
    assert any("final_object_count" in failure for failure in result.failures)


def test_quality_gate_rejects_weak_drop() -> None:
    result = evaluate_quality_gate(
        {"score": 0.9, "final_object_count": 20, "drop_impact_score": 0.1},
        QualityGateThresholds(min_drop_impact=0.5, min_object_count=1),
    )

    assert result.passed is False
    assert any("drop_impact" in failure for failure in result.failures)


def test_quality_gate_accepts_good_candidate() -> None:
    result = evaluate_quality_gate(
        {
            "score": 0.9,
            "final_object_count": 40,
            "removed_object_ratio": 0.1,
            "drop_impact_score": 0.8,
            "metrics": {"density_alignment_score": 0.8},
            "score_breakdown": {"editor_validity": 0.9, "playability_safety": 0.9},
        },
        QualityGateThresholds(min_object_count=10),
    )

    assert result.passed is True


def test_quality_gate_report_lists_failures() -> None:
    result = evaluate_quality_gate({"score": 0.1, "final_object_count": 0})

    assert result.failures
    assert result.to_dict()["passed"] is False
