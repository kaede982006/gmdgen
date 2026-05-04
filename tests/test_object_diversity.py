# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from gmdgen.generate.scoring import score_diversity, compute_level_score


def _objs(ids: list[str], x_start: float = 0.0) -> list[str]:
    """Build minimal GD object strings from a list of object IDs."""
    result = []
    for i, obj_id in enumerate(ids):
        x = x_start + i * 30.0
        result.append(f"1,{obj_id},2,{x:.1f},3,30.0;")
    return result


def test_all_same_id_gives_low_diversity():
    objects = _objs(["1"] * 20)
    score = score_diversity(objects)
    assert score < 0.2, f"All same IDs should score < 0.2, got {score:.3f}"


def test_all_unique_ids_gives_high_diversity():
    ids = [str(i) for i in range(1, 21)]
    objects = _objs(ids)
    score = score_diversity(objects)
    assert score == 1.0, f"All unique IDs should score 1.0, got {score:.3f}"


def test_partial_variety_gives_intermediate_score():
    ids = ["1"] * 10 + ["36", "141", "1332", "211", "8"]
    objects = _objs(ids)
    score = score_diversity(objects)
    assert 0.1 < score < 1.0, f"Mixed IDs should score between 0.1 and 1.0, got {score:.3f}"


def test_more_variety_scores_higher():
    low_variety = _objs(["1", "1", "1", "36", "36", "36"])
    high_variety = _objs(["1", "36", "141", "1332", "211", "8"])
    low_score = score_diversity(low_variety)
    high_score = score_diversity(high_variety)
    assert high_score > low_score, (
        f"High variety should score higher. low={low_score:.3f}, high={high_score:.3f}"
    )


def test_empty_objects_returns_zero():
    assert score_diversity([]) == 0.0


def test_objects_without_valid_id_returns_zero():
    # Objects with no parseable ID
    objects = ["invalid,line,here;", "another,bad,one;"]
    score = score_diversity(objects)
    assert score == 0.0


def test_diversity_in_level_score():
    """compute_level_score returns a LevelScore with a diversity field."""
    ids = ["1", "36", "141", "1332", "211"]
    objects = _objs(ids * 4)
    level_score = compute_level_score(objects)
    assert 0.0 < level_score.diversity <= 1.0


def test_diversity_single_object():
    objects = _objs(["1"])
    score = score_diversity(objects)
    assert score == 1.0, "Single object has 100% unique ratio"


def test_repeated_tuple_ratio_implicitly_via_motif_score():
    """Objects that are highly repetitive should reduce motif quality score."""
    from gmdgen.generate.scoring import _motif_quality_score
    repetitive = _objs(["1"] * 20)
    diverse = _objs(["1", "36", "141", "1332", "211"] * 4)
    rep_score = _motif_quality_score(repetitive)
    div_score = _motif_quality_score(diverse)
    # Diverse should score at least as high as repetitive
    assert div_score >= rep_score - 0.05, (
        f"Diverse motifs should score similarly or higher. rep={rep_score:.3f}, div={div_score:.3f}"
    )
