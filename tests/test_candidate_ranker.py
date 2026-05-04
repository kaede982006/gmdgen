# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from gmdgen.generate.candidate_ranker import CandidateRanker


def test_ranker_prefers_playable_candidate() -> None:
    ranker = CandidateRanker()
    c1 = {"metrics": {"playability_safety_score": 0.2, "repair_loss_ratio": 0.0}}
    c2 = {"metrics": {"playability_safety_score": 0.9, "repair_loss_ratio": 0.0}}
    ranked = ranker.rank_candidates([c1, c2])
    assert ranker.extract_candidate_features(ranked[0]).playability == 0.9

def test_ranker_penalizes_repair_loss() -> None:
    ranker = CandidateRanker()
    c1 = {"metrics": {"playability_safety_score": 0.9, "repair_loss_ratio": 0.5}}
    c2 = {"metrics": {"playability_safety_score": 0.9, "repair_loss_ratio": 0.1}}
    ranked = ranker.rank_candidates([c1, c2])
    assert ranker.extract_candidate_features(ranked[0]).repair_loss == 0.1

def test_ranker_rejects_garbage_description() -> None:
    ranker = CandidateRanker()
    c1 = {"metrics": {"playability_safety_score": 0.9, "repair_loss_ratio": 0.9}} # > 0.8 is hard reject
    c2 = {"metrics": {"playability_safety_score": 0.5, "repair_loss_ratio": 0.1}}
    ranked = ranker.rank_candidates([c1, c2])
    assert ranker.extract_candidate_features(ranked[0]).repair_loss == 0.1
