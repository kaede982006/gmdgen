# SPDX-License-Identifier: GPL-3.0-or-later
from dataclasses import dataclass, field
from typing import Any

@dataclass
class CandidateFeatures:
    object_count: int = 0
    trigger_count: int = 0
    density_alignment: float = 0.0
    drop_impact: float = 0.0
    beat_sync: float = 0.0
    onset_sync: float = 0.0
    repair_loss: float = 0.0
    object_diversity: float = 0.0
    motif_match: float = 0.0
    style_match: float = 0.0
    editor_safety: float = 0.0
    playability_safety: float = 0.0
    geode_penalty: float = 0.0
    description_quality: float = 0.0
    critic_score: float = 0.0
    user_preference_prior: float = 0.0

class CandidateRanker:
    @classmethod
    def extract_features(cls, candidate: dict[str, Any]) -> CandidateFeatures:
        report = candidate.get("validation_report", {})
        metrics = report.get("metrics", {})
        return CandidateFeatures(
            object_count=report.get("final_object_count", 0),
            trigger_count=report.get("trigger_count", 0),
            density_alignment=metrics.get("density_alignment_score", 0.0),
            drop_impact=metrics.get("drop_impact_score", 0.0),
            beat_sync=metrics.get("beat_sync_score", 0.0),
            onset_sync=metrics.get("onset_sync_score", 0.0),
            repair_loss=metrics.get("repair_loss_ratio", 0.0),
            object_diversity=metrics.get("object_diversity", 0.0),
            motif_match=metrics.get("motif_match", 0.0),
            style_match=metrics.get("style_match", 0.0),
            editor_safety=metrics.get("editor_safety_score", 0.0),
            playability_safety=metrics.get("playability_safety_score", 0.0),
            geode_penalty=metrics.get("geode_penalty", 0.0),
            description_quality=metrics.get("description_quality", 1.0),
            critic_score=metrics.get("critic_score", 0.0),
            user_preference_prior=metrics.get("user_preference_prior", 0.0),
        )

    @classmethod
    def rank_candidates(cls, candidates: list[dict[str, Any]], weights: dict[str, float] | None = None) -> list[dict[str, Any]]:
        default_weights = {
            "density_alignment": 1.0,
            "drop_impact": 1.5,
            "beat_sync": 1.0,
            "onset_sync": 1.0,
            "repair_loss": -2.0,
            "style_match": 1.0,
            "editor_safety": 2.0,
            "geode_penalty": -2.0,
            "critic_score": 1.5,
        }
        w = weights or default_weights

        def compute_score(features: CandidateFeatures) -> float:
            score = 0.0
            score += features.density_alignment * w.get("density_alignment", 0.0)
            score += features.drop_impact * w.get("drop_impact", 0.0)
            score += features.beat_sync * w.get("beat_sync", 0.0)
            score += features.onset_sync * w.get("onset_sync", 0.0)
            score += features.repair_loss * w.get("repair_loss", 0.0)
            score += features.style_match * w.get("style_match", 0.0)
            score += features.editor_safety * w.get("editor_safety", 0.0)
            score += features.geode_penalty * w.get("geode_penalty", 0.0)
            score += features.critic_score * w.get("critic_score", 0.0)
            return score

        scored_candidates = []
        for cand in candidates:
            features = cls.extract_features(cand)
            score = compute_score(features)
            cand["ml_ranking_score"] = score
            cand["ml_features"] = features.__dict__
            scored_candidates.append(cand)

        return sorted(scored_candidates, key=lambda c: c["ml_ranking_score"], reverse=True)
