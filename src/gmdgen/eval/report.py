from dataclasses import dataclass, field
from typing import Any

@dataclass
class EvalResult:
    case_id: str
    passed: bool
    score_breakdown: dict[str, float] = field(default_factory=dict)
    failed_thresholds: list[str] = field(default_factory=list)
    quality_loss_reasons: list[str] = field(default_factory=list)
    selected_candidate_summary: dict[str, Any] = field(default_factory=dict)
    plan_snapshot_summary: dict[str, Any] = field(default_factory=dict)
    learning_example_id: str = ""
    report_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "score_breakdown": self.score_breakdown,
            "failed_thresholds": self.failed_thresholds,
            "quality_loss_reasons": self.quality_loss_reasons,
            "selected_candidate_summary": self.selected_candidate_summary,
            "plan_snapshot_summary": self.plan_snapshot_summary,
            "learning_example_id": self.learning_example_id,
            "report_path": self.report_path,
        }
