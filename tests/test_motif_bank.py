from __future__ import annotations

from gmdgen.eval.motif_bank import MotifBank, MotifQualityScore


def test_motif_quality_score_computed() -> None:
    score = MotifQualityScore(density_score=1.0, structure_score=1.0)
    assert score.total > 0.0

def test_motif_bank_serializes() -> None:
    motif = MotifBank(motif_id="test1", density=0.5)
    data = motif.to_dict()
    assert data["motif_id"] == "test1"
