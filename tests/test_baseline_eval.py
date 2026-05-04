# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
from pathlib import Path

from gmdgen.eval.baseline import BaselineEvalReport, compare_eval_reports, save_baseline_report


def test_baseline_eval_report_serializes() -> None:
    report = BaselineEvalReport(
        model_name="test_model",
        average_score=0.75,
        quality_gate_pass_rate=0.8,
    )
    data = report.to_dict()
    assert data["model_name"] == "test_model"
    assert data["average_score"] == 0.75

    report2 = BaselineEvalReport.from_dict(data)
    assert report2.model_name == "test_model"


def test_eval_baseline_can_compare_two_reports() -> None:
    before = BaselineEvalReport(average_score=0.5, quality_gate_pass_rate=0.5)
    after = BaselineEvalReport(average_score=0.8, quality_gate_pass_rate=0.9)
    result = compare_eval_reports(before, after)
    assert result["score_improvement"] > 0
    assert result["overall_improved"] is True
