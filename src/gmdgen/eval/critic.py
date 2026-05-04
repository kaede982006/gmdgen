from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CriticOutput:
    overall_quality_score: float = 0.0
    gd_style_score: float = 0.0
    rhythm_score: float = 0.0
    playability_score: float = 0.0
    drop_impact_score: float = 0.0
    visual_interest_score: float = 0.0
    repetition_score: float = 0.0
    trigger_safety_score: float = 0.0
    top_weaknesses: list[str] = field(default_factory=list)
    concrete_revision_instructions: list[str] = field(default_factory=list)
    regenerate_sections: list[int] = field(default_factory=list)
    reduce_density_sections: list[int] = field(default_factory=list)
    increase_density_sections: list[int] = field(default_factory=list)
    trigger_fix_instructions: list[str] = field(default_factory=list)
    playability_fix_instructions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_quality_score": self.overall_quality_score,
            "gd_style_score": self.gd_style_score,
            "rhythm_score": self.rhythm_score,
            "playability_score": self.playability_score,
            "drop_impact_score": self.drop_impact_score,
            "visual_interest_score": self.visual_interest_score,
            "repetition_score": self.repetition_score,
            "trigger_safety_score": self.trigger_safety_score,
            "top_weaknesses": list(self.top_weaknesses),
            "concrete_revision_instructions": list(self.concrete_revision_instructions),
            "regenerate_sections": list(self.regenerate_sections),
            "reduce_density_sections": list(self.reduce_density_sections),
            "increase_density_sections": list(self.increase_density_sections),
            "trigger_fix_instructions": list(self.trigger_fix_instructions),
            "playability_fix_instructions": list(self.playability_fix_instructions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CriticOutput:
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


def run_critic_evaluation(candidate_report: dict[str, Any]) -> CriticOutput:
    # In a real scenario, this would format a prompt with:
    # - candidate summary
    # - plan snapshot diff
    # - repair loss breakdown
    # - playability breakdown
    # - QualityGate failed checks
    # - motif match report
    # - section density report
    # And then call the AI API.
    # For now, return a mock output.
    
    output = CriticOutput()
    metrics = candidate_report.get("metrics", {})
    output.overall_quality_score = float(metrics.get("final_score", 0.5))
    
    if float(metrics.get("repair_loss_ratio", 0.0)) > 0.4:
        output.top_weaknesses.append("High repair loss")
        output.concrete_revision_instructions.append("Reduce overlapping objects and check trigger bounds.")
        
    if float(metrics.get("playability_safety_score", 1.0)) < 0.6:
        output.top_weaknesses.append("Poor playability")
        output.playability_fix_instructions.append("Increase gap between jump orbs and remove blind transitions.")
        
    return output
