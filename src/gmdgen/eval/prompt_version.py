# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from gmdgen.eval.baseline import BaselineEvalReport


@dataclass(slots=True)
class PromptVersion:
    version_id: str = ""
    name: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    purpose: str = ""
    system_prompt_hash: str = ""
    planner_prompt_hash: str = ""
    trigger_prompt_hash: str = ""
    critic_prompt_hash: str = ""
    revision_prompt_hash: str = ""
    quality_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "name": self.name,
            "created_at": self.created_at,
            "purpose": self.purpose,
            "system_prompt_hash": self.system_prompt_hash,
            "planner_prompt_hash": self.planner_prompt_hash,
            "trigger_prompt_hash": self.trigger_prompt_hash,
            "critic_prompt_hash": self.critic_prompt_hash,
            "revision_prompt_hash": self.revision_prompt_hash,
            "quality_notes": self.quality_notes,
        }


def register_prompt_version(version: PromptVersion, registry_dir: Path) -> None:
    registry_dir.mkdir(parents=True, exist_ok=True)
    path = registry_dir / f"{version.version_id}.json"
    path.write_text(json.dumps(version.to_dict(), indent=2), encoding="utf-8")


def select_prompt_version(version_id: str, registry_dir: Path) -> PromptVersion | None:
    path = registry_dir / f"{version_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return PromptVersion(**{k: v for k, v in data.items() if hasattr(PromptVersion, k)})


def run_prompt_ab_eval(version_a: str, version_b: str, registry_dir: Path) -> dict[str, Any]:
    # Mock evaluation for A/B testing
    return {
        "version_a": version_a,
        "version_b": version_b,
        "comparison": {
            "score_improvement": 0.05,
            "playability_improvement": 0.02,
        },
        "winner": version_b,
    }


def compare_prompt_eval_results(report_a: BaselineEvalReport, report_b: BaselineEvalReport) -> dict[str, Any]:
    return {
        "prompt_a": report_a.prompt_version,
        "prompt_b": report_b.prompt_version,
        "score_diff": report_b.average_score - report_a.average_score,
        "pass_rate_diff": report_b.quality_gate_pass_rate - report_a.quality_gate_pass_rate,
    }


def save_prompt_eval_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
