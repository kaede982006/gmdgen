# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from gmdgen.eval.critic import CriticOutput, run_critic_evaluation


def test_critic_output_schema_valid() -> None:
    output = CriticOutput()
    assert isinstance(output.to_dict(), dict)

def test_critic_revision_instructions_actionable() -> None:
    candidate_report = {
        "metrics": {
            "repair_loss_ratio": 0.5,
            "playability_safety_score": 0.4,
            "final_score": 0.3
        }
    }
    critic = run_critic_evaluation(candidate_report)
    assert "High repair loss" in critic.top_weaknesses
    assert len(critic.concrete_revision_instructions) > 0
    assert len(critic.playability_fix_instructions) > 0
