# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from pathlib import Path

from gmdgen.generate.eval_suite import EvalCase, compare_eval_results, run_generation_eval_suite, save_eval_report


def test_eval_case_runs() -> None:
    cases = [EvalCase(case_id="steady", expected_properties={"min_object_count": 2})]

    results = run_generation_eval_suite(cases, lambda _case: {"validation_report": {"final_object_count": 4, "score": 0.8}})

    assert results[0].passed is True


def test_eval_fails_empty_drop() -> None:
    cases = [EvalCase(case_id="drop", min_score_thresholds={"drop_impact": 0.5})]

    results = run_generation_eval_suite(cases, lambda _case: {"validation_report": {"drop_impact_score": 0.1, "score": 0.5}})

    assert results[0].passed is False


def test_eval_report_serializes(tmp_path: Path) -> None:
    results = run_generation_eval_suite([EvalCase(case_id="a")], lambda _case: {"validation_report": {"score": 0.7}})
    path = tmp_path / "eval.json"

    save_eval_report(results, path)

    assert path.exists()
    assert "summary" in path.read_text(encoding="utf-8")


def test_eval_thresholds_applied() -> None:
    cases = [EvalCase(case_id="threshold", expected_properties={"max_repair_loss": 0.2})]
    results = run_generation_eval_suite(cases, lambda _case: {"validation_report": {"removed_object_ratio": 0.5}})

    assert "max_repair_loss_exceeded" in results[0].failures
    assert compare_eval_results(results)["failed"] == 1
