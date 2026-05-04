# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass(slots=True)
class EvalCase:
    case_id: str
    synthetic_audio: str = ""
    reference_level: str = ""
    difficulty: str = "normal"
    target_style: str = ""
    expected_properties: dict[str, Any] = field(default_factory=dict)
    min_score_thresholds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvalResult:
    case_id: str
    passed: bool
    score: float = 0.0
    failures: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_generation_eval_suite(
    cases: list[EvalCase],
    generate_case: Callable[[EvalCase], dict[str, Any]],
) -> list[EvalResult]:
    results = []
    for case in cases:
        output = generate_case(case)
        report = output.get("validation_report", output)
        results.append(_evaluate_case(case, report if isinstance(report, dict) else {}))
    return results


def compare_eval_results(results: list[EvalResult]) -> dict[str, Any]:
    passed = sum(1 for result in results if result.passed)
    return {
        "case_count": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "average_score": sum(result.score for result in results) / max(1, len(results)),
    }


def save_eval_report(results: list[EvalResult], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": compare_eval_results(results),
        "results": [result.to_dict() for result in results],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _evaluate_case(case: EvalCase, report: dict[str, Any]) -> EvalResult:
    failures: list[str] = []
    metrics = {
        "object_count": int(report.get("final_object_count", 0) or 0),
        "trigger_count": int(report.get("generated_trigger_count", report.get("final_trigger_count", 0)) or 0),
        "repair_loss": max(float(report.get("removed_object_ratio", 0.0) or 0.0), float(report.get("removed_trigger_ratio", 0.0) or 0.0)),
        "drop_impact": float(report.get("drop_impact_score", 0.0) or 0.0),
        "density_alignment": float((report.get("metrics", {}) or {}).get("density_alignment_score", 0.0) or 0.0),
        "object_diversity": float((report.get("metrics", {}) or {}).get("object_diversity_score", 0.0) or 0.0),
        "score": float(report.get("score", 0.0) or 0.0),
    }
    expected = case.expected_properties
    if metrics["object_count"] < int(expected.get("min_object_count", 0)):
        failures.append("min_object_count_not_met")
    if metrics["trigger_count"] < int(expected.get("min_trigger_count", 0)):
        failures.append("min_trigger_count_not_met")
    if metrics["repair_loss"] > float(expected.get("max_repair_loss", 1.0)):
        failures.append("max_repair_loss_exceeded")
    for name, threshold in case.min_score_thresholds.items():
        if float(metrics.get(name, 0.0)) < float(threshold):
            failures.append(f"{name}_below_threshold")
    return EvalResult(case_id=case.case_id, passed=not failures, score=metrics["score"], failures=failures, metrics=metrics)
