# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CandidateFeatures:
    object_count: int = 0
    trigger_count: int = 0
    rendered_object_count: int = 0
    rendered_trigger_count: int = 0
    repair_loss: float = 0.0
    playability: float = 0.0
    editor_safety: float = 0.0
    trigger_validity: float = 0.0
    group_validity: float = 0.0
    beat_sync: float = 0.0
    onset_sync: float = 0.0
    density_alignment: float = 0.0
    drop_impact: float = 0.0
    object_diversity: float = 0.0
    style_match: float = 0.0
    motif_match: float = 0.0
    description_quality: float = 0.0
    geode_score: float = 0.0
    critic_score: float = 0.0
    user_preference_prior: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "object_count": float(self.object_count),
            "trigger_count": float(self.trigger_count),
            "rendered_object_count": float(self.rendered_object_count),
            "rendered_trigger_count": float(self.rendered_trigger_count),
            "repair_loss": self.repair_loss,
            "playability": self.playability,
            "editor_safety": self.editor_safety,
            "trigger_validity": self.trigger_validity,
            "group_validity": self.group_validity,
            "beat_sync": self.beat_sync,
            "onset_sync": self.onset_sync,
            "density_alignment": self.density_alignment,
            "drop_impact": self.drop_impact,
            "object_diversity": self.object_diversity,
            "style_match": self.style_match,
            "motif_match": self.motif_match,
            "description_quality": self.description_quality,
            "geode_score": self.geode_score,
            "critic_score": self.critic_score,
            "user_preference_prior": self.user_preference_prior,
        }


class CandidateRanker:
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or {
            "playability": 1.5,
            "repair_loss": -2.0,
            "editor_safety": 1.0,
            "drop_impact": 1.2,
            "beat_sync": 1.0,
            "trigger_validity": 1.0,
        }

    def extract_candidate_features(self, report: dict[str, Any]) -> CandidateFeatures:
        features = CandidateFeatures()
        metrics = report.get("metrics", {})
        
        # Simple extraction mapping
        features.playability = float(metrics.get("playability_safety_score", 0.0))
        features.repair_loss = float(metrics.get("repair_loss_ratio", 0.0))
        features.editor_safety = float(metrics.get("editor_safety_score", 0.0))
        features.drop_impact = float(metrics.get("drop_impact_score", 0.0))
        features.density_alignment = float(metrics.get("density_alignment_score", 0.0))
        features.object_diversity = float(metrics.get("object_diversity_score", 0.0))
        
        plan_count = report.get("plan_count_report", {})
        if isinstance(plan_count, dict):
            features.object_count = int(plan_count.get("raw_ai_objects") or 0)
            features.trigger_count = int(plan_count.get("raw_ai_triggers") or 0)
            
        return features

    def rank_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []
            
        def _score(c: dict[str, Any]) -> float:
            features = self.extract_candidate_features(c)
            # Hard reject checks
            if features.repair_loss > 0.8:
                return -100.0
            
            score = 0.0
            for k, w in self.weights.items():
                val = getattr(features, k, 0.0)
                score += val * w
            return score

        return sorted(candidates, key=_score, reverse=True)

    def explain_candidate_ranking(self, candidate: dict[str, Any]) -> str:
        features = self.extract_candidate_features(candidate)
        return f"Ranked based on Playability={features.playability:.2f}, RepairLoss={features.repair_loss:.2f}"

    def calibrate_ranker_from_feedback(self, feedback_records: list[dict[str, Any]]) -> None:
        pass

    def export_ranker_training_data(self, output_path: str) -> None:
        pass
