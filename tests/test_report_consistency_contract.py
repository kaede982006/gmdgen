# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import pytest

from gmdgen.validation.report_consistency import (
    ReportConsistencyError,
    validate_generation_report_consistency,
)


def _base_report() -> dict:
    return {
        "planner_status": "ollama_used",
        "planner_fallback_used": False,
        "raw_objects": 4,
        "candidate_ir_objects": 4,
        "candidate_objects": 4,
        "parsed_candidate_objects": 4,
        "serialized_objects": 4,
        "final_objects": 4,
        "syntax_validation": {"passed": True},
        "semantic_validation": {"passed": True},
        "playability_validation": {"passed": True},
        "repair_applied": False,
        "repair_loss": 0.0,
        "quality_gate_passed": True,
        "low_quality_draft_saved": False,
        "final_success": True,
    }


def test_raw_candidate_final_count_mismatch_is_hard_failure() -> None:
    report = _base_report()
    report["candidate_objects"] = 5

    with pytest.raises(ReportConsistencyError):
        validate_generation_report_consistency(report, hard=True)


def test_missing_target_group_is_hard_failure() -> None:
    report = _base_report()
    report["missing_target_group"] = 1

    result = validate_generation_report_consistency(report)

    assert result.passed is False
    assert "missing_target_group_positive" in result.errors


def test_low_quality_draft_cannot_be_final_success() -> None:
    report = _base_report()
    report["low_quality_draft_saved"] = True

    result = validate_generation_report_consistency(report)

    assert result.passed is False
    assert "low_quality_draft_marked_final_success" in result.errors


def test_planner_fallback_cannot_be_normal_final_success() -> None:
    report = _base_report()
    report["planner_status"] = "fallback"
    report["planner_fallback_used"] = True

    result = validate_generation_report_consistency(report)

    assert result.passed is False
    assert "planner_fallback_marked_final_success" in result.errors


def test_repair_requires_updated_metrics() -> None:
    report = _base_report()
    report["repair_applied"] = True
    report["repair_metrics_updated"] = False

    result = validate_generation_report_consistency(report)

    assert result.passed is False
    assert "repair_metrics_not_updated_after_repair" in result.errors
