# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import pytest
from gmdgen.generate.quality_gate import QualityGateThresholds, evaluate_quality_gate


def _good_report(**overrides) -> dict:
    base = {
        "score": 0.9,
        "final_object_count": 40,
        "removed_object_ratio": 0.05,
        "removed_trigger_ratio": 0.02,
        "drop_impact_score": 0.8,
        "metrics": {"density_alignment_score": 0.8},
        "score_breakdown": {"editor_validity": 0.9, "playability_safety": 0.9},
    }
    base.update(overrides)
    return base


def test_high_repair_loss_rejected():
    report = _good_report(removed_object_ratio=0.8)
    result = evaluate_quality_gate(report, QualityGateThresholds(max_repair_loss=0.5))
    assert result.passed is False
    assert any("repair_loss" in f for f in result.failures)


def test_repair_loss_exactly_at_threshold_passes():
    report = _good_report(removed_object_ratio=0.5)
    result = evaluate_quality_gate(report, QualityGateThresholds(max_repair_loss=0.5))
    # At threshold: not strictly above, so should pass this check
    assert not any("repair_loss" in f for f in result.failures)


def test_repair_loss_just_above_threshold_fails():
    report = _good_report(removed_object_ratio=0.51)
    result = evaluate_quality_gate(report, QualityGateThresholds(max_repair_loss=0.5))
    assert result.passed is False
    assert any("repair_loss" in f for f in result.failures)


def test_low_repair_loss_passes():
    report = _good_report(removed_object_ratio=0.05)
    result = evaluate_quality_gate(report, QualityGateThresholds(max_repair_loss=0.5, min_object_count=1))
    assert result.passed is True


def test_trigger_repair_loss_also_checked():
    # removed_trigger_ratio is also checked (max of object/trigger)
    report = _good_report(removed_object_ratio=0.1, removed_trigger_ratio=0.9)
    result = evaluate_quality_gate(report, QualityGateThresholds(max_repair_loss=0.5))
    assert result.passed is False
    assert any("repair_loss" in f for f in result.failures)


def test_repair_loss_breakdown_captured_in_result():
    breakdown = {"x_mono_fixes": 500, "spacing_fixes": 100}
    report = _good_report(removed_object_ratio=0.7, repair_loss_breakdown=breakdown)
    result = evaluate_quality_gate(report, QualityGateThresholds(max_repair_loss=0.5, min_object_count=1))
    assert result.passed is False
    assert result.repair_loss_breakdown == breakdown


def test_recommended_actions_given_on_high_repair_loss():
    report = _good_report(removed_object_ratio=0.8)
    result = evaluate_quality_gate(report, QualityGateThresholds(max_repair_loss=0.5, min_object_count=1))
    assert result.recommended_actions, "Expected recommended actions when repair_loss is high"


def test_best_candidate_selection_prefers_lower_repair_loss():
    """Simulate choosing between two candidates via quality gate."""
    bad = _good_report(removed_object_ratio=0.7, score=0.8)
    good = _good_report(removed_object_ratio=0.1, score=0.6)

    bad_result = evaluate_quality_gate(bad, QualityGateThresholds(max_repair_loss=0.5, min_object_count=1))
    good_result = evaluate_quality_gate(good, QualityGateThresholds(max_repair_loss=0.5, min_object_count=1))

    assert bad_result.passed is False
    assert good_result.passed is True


def test_zero_repair_loss_report_passes_guard():
    report = _good_report(removed_object_ratio=0.0, removed_trigger_ratio=0.0)
    result = evaluate_quality_gate(report, QualityGateThresholds(max_repair_loss=0.5, min_object_count=1))
    assert not any("repair_loss" in f for f in result.failures)


def test_quality_gate_result_serializes_to_dict():
    report = _good_report(removed_object_ratio=0.6)
    result = evaluate_quality_gate(report)
    d = result.to_dict()
    assert "passed" in d
    assert "failures" in d
    assert "metrics" in d
    assert "recommended_actions" in d
