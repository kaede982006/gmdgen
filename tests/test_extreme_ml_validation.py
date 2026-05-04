from __future__ import annotations

from pathlib import Path

from gmdgen.eval.extreme_validation import ExtremeMLValidationReport, run_extreme_ml_validation, generate_model_improvement_report


def test_extreme_ml_validation_report_serializes() -> None:
    report = ExtremeMLValidationReport(baseline_score=0.9, best_prompt_version="v2")
    data = report.to_dict()
    assert data["baseline_score"] == 0.9
    
    report2 = ExtremeMLValidationReport.from_dict(data)
    assert report2.best_prompt_version == "v2"


def test_extreme_ml_validation_runs_fake_suite(tmp_path: Path) -> None:
    report = run_extreme_ml_validation(tmp_path)
    assert report.baseline_score == 0.5
    assert report.dataset_health == "Good"


def test_compare_eval_reports_detects_improvement() -> None:
    r1 = ExtremeMLValidationReport(baseline_score=0.5)
    r2 = ExtremeMLValidationReport(baseline_score=0.8)
    diff = generate_model_improvement_report(r1, r2)
    assert diff["baseline_score_improvement"] > 0
