# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class BaselineEvalReport:
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    dataset_hash: str = ""
    model_name: str = ""
    quality_mode: str = "Extreme"
    eval_cases: int = 0
    average_score: float = 0.0
    average_playability: float = 0.0
    average_drop_impact: float = 0.0
    average_density_alignment: float = 0.0
    average_object_diversity: float = 0.0
    average_trigger_validity: float = 0.0
    average_repair_loss: float = 0.0
    quality_gate_pass_rate: float = 0.0
    draft_rate: float = 0.0
    failure_reason_distribution: dict[str, int] = field(default_factory=dict)
    prompt_version: str = "v1"
    renderer_version: str = "v1"
    scoring_version: str = "v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "dataset_hash": self.dataset_hash,
            "model_name": self.model_name,
            "quality_mode": self.quality_mode,
            "eval_cases": self.eval_cases,
            "average_score": self.average_score,
            "average_playability": self.average_playability,
            "average_drop_impact": self.average_drop_impact,
            "average_density_alignment": self.average_density_alignment,
            "average_object_diversity": self.average_object_diversity,
            "average_trigger_validity": self.average_trigger_validity,
            "average_repair_loss": self.average_repair_loss,
            "quality_gate_pass_rate": self.quality_gate_pass_rate,
            "draft_rate": self.draft_rate,
            "failure_reason_distribution": dict(self.failure_reason_distribution),
            "prompt_version": self.prompt_version,
            "renderer_version": self.renderer_version,
            "scoring_version": self.scoring_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineEvalReport:
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


def save_baseline_report(report: BaselineEvalReport, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"baseline_eval_{report.created_at.replace(':', '')}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path


def compare_eval_reports(before: BaselineEvalReport, after: BaselineEvalReport) -> dict[str, Any]:
    return {
        "score_improvement": after.average_score - before.average_score,
        "playability_improvement": after.average_playability - before.average_playability,
        "repair_loss_reduction": before.average_repair_loss - after.average_repair_loss,
        "quality_gate_pass_rate_improvement": after.quality_gate_pass_rate - before.quality_gate_pass_rate,
        "trigger_validity_improvement": after.average_trigger_validity - before.average_trigger_validity,
        "overall_improved": (
            after.average_score > before.average_score
            and after.quality_gate_pass_rate >= before.quality_gate_pass_rate
        ),
    }
