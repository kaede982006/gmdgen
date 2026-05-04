# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from gmdgen.generate.scoring import (
    LevelScore,
    ScoringConfig,
    ScoreBreakdown,
    compute_audio_conditioned_score,
    compute_level_score,
    score_density,
    score_diversity,
    score_object_class,
    score_position,
    score_richness,
    score_trigger_integrity,
    score_visible,
    validation_report_to_dict,
    DEFAULT_WEIGHTS,
)
from gmdgen.gd.plans import ValidationReport
from gmdgen.gd.plans import SectionPlan
from gmdgen.gd.time_mapping import SpeedState


def _obj(object_id: str, x: int, y: int) -> str:
    return f"1,{object_id},2,{x},3,{y}"


def _trigger(x: int, target: int) -> str:
    return f"1,29,2,{x},3,180,51,{target}"


def _group_obj(object_id: str, x: int, group: int) -> str:
    return f"1,{object_id},2,{x},3,180,155,{group}"


# ── L_position ────────────────────────────────────────────────────────────────

def test_score_position_perfect_monotone() -> None:
    objects = [_obj("1", 30, 180), _obj("1", 60, 180), _obj("1", 90, 180)]
    assert score_position(objects) == 1.0


def test_score_position_fully_reversed() -> None:
    objects = [_obj("1", 90, 180), _obj("1", 60, 180), _obj("1", 30, 180)]
    assert score_position(objects) == 0.0


def test_score_position_empty() -> None:
    assert score_position([]) == 1.0


# ── L_density ─────────────────────────────────────────────────────────────────

def test_score_density_uniform_objects() -> None:
    objects = [_obj("1", i * 30, 180) for i in range(1, 11)]
    score = score_density(objects, grid_unit=30)
    assert 0.0 <= score <= 1.0


def test_score_density_clustered_objects() -> None:
    objects = [_obj("1", 30, 180)] * 20  # all at x=30
    score_sparse = score_density([_obj("1", i * 90, 180) for i in range(1, 6)], grid_unit=30)
    score_clustered = score_density(objects, grid_unit=30)
    assert score_sparse >= score_clustered  # uniform ≥ clustered


# ── L_object_class ────────────────────────────────────────────────────────────

def test_score_object_class_no_reference() -> None:
    objects = [_obj("1", 30, 180), _obj("500", 60, 180)]
    assert score_object_class(objects) == 1.0


def test_score_object_class_perfect_match() -> None:
    objects = [_obj("1", 30, 180), _obj("500", 60, 180)]  # structure + decoration
    from gmdgen.representation.object_classifier import classify
    counts = {}
    for obj in objects:
        from gmdgen.features.tokenizer import extract_object_id
        oid = extract_object_id(obj)
        if oid:
            cls = classify(oid).value
            counts[cls] = counts.get(cls, 0) + 1
    total = sum(counts.values())
    target = {k: v / total for k, v in counts.items()}
    assert score_object_class(objects, target_distribution=target) == 1.0


# ── L_trigger ─────────────────────────────────────────────────────────────────

def test_score_trigger_all_valid() -> None:
    objects = [
        _group_obj("1", 10, group=5),
        _trigger(20, target=5),
    ]
    assert score_trigger_integrity(objects) == 1.0


def test_score_trigger_all_orphan() -> None:
    objects = [
        _trigger(10, target=99),  # group 99 not defined
    ]
    assert score_trigger_integrity(objects) == 0.0


def test_score_trigger_no_triggers() -> None:
    objects = [_obj("1", 30, 180)]
    assert score_trigger_integrity(objects) == 1.0


# ── L_richness ────────────────────────────────────────────────────────────────

def test_score_richness_all_rich() -> None:
    objects = ["1,1,2,30,3,180,4,1,155,5,6,90"]  # > 6 fields
    assert score_richness(objects) == 1.0


def test_score_richness_all_simple() -> None:
    objects = [_obj("1", 30, 180)]  # exactly 6 fields
    assert score_richness(objects) == 0.0


# ── L_diversity ───────────────────────────────────────────────────────────────

def test_score_diversity_all_unique() -> None:
    objects = [_obj(str(i), i * 30, 180) for i in range(1, 6)]
    assert score_diversity(objects) == 1.0


def test_score_diversity_all_same() -> None:
    objects = [_obj("1", 30, 180)] * 5
    assert score_diversity(objects) < 0.5


# ── L_visible ─────────────────────────────────────────────────────────────────

def test_score_visible_all_visible() -> None:
    objects = [_obj("1", 30, 180), _obj("500", 60, 180)]
    assert score_visible(objects) == 1.0


def test_score_visible_triggers_not_visible() -> None:
    triggers = [f"1,29,2,{i*10},3,180" for i in range(5)]
    assert score_visible(triggers) == 0.0


# ── compute_level_score ───────────────────────────────────────────────────────

def test_compute_level_score_returns_level_score() -> None:
    objects = [
        _group_obj("1", 30, group=1),
        _obj("500", 60, 180),
        _trigger(90, target=1),
    ]
    score = compute_level_score(objects)
    assert isinstance(score, LevelScore)
    assert score.total > 0.0
    all_terms = score.to_dict()
    for key in ["position", "trigger", "richness", "diversity", "visible"]:
        assert key in all_terms
        assert 0.0 <= all_terms[key] <= 1.0


def test_compute_level_score_empty_no_objects() -> None:
    # Empty list → diversity=0 and richness=0, but position/trigger/class default to 1.0
    # so total is nonzero. Just verify it doesn't raise and total is a float.
    score = compute_level_score([])
    assert isinstance(score.total, float)


class _AudioFeatures:
    beat_times = [0.0, 0.5, 1.0, 1.5]
    onset_times = [0.0, 0.5, 1.0, 1.5]
    sections = []


def test_score_breakdown_fields_present() -> None:
    score = compute_audio_conditioned_score(
        [_obj("1", 0, 180), _obj("1", 155, 180)],
        audio_features=_AudioFeatures(),
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=10,
    )
    assert isinstance(score, ScoreBreakdown)
    data = score.to_dict()
    for key in ["beat_sync", "onset_sync", "section_alignment", "playability_safety", "total"]:
        assert key in data


def test_scores_are_normalized() -> None:
    score = compute_audio_conditioned_score(
        [_obj("1", 0, 180), _obj("1006", 155, 240)],
        audio_features=_AudioFeatures(),
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=10,
    ).to_dict()
    for key, value in score.items():
        if key == "total":
            assert value >= 0.0
        else:
            assert 0.0 <= value <= 1.0


def test_beat_sync_scores_important_events_not_all_beats() -> None:
    on_beat = compute_audio_conditioned_score(
        [_obj("1", 0, 180), _obj("500", 77, 240)],
        audio_features=_AudioFeatures(),
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=10,
    )
    off_beat = compute_audio_conditioned_score(
        [_obj("1", 77, 180), _obj("500", 0, 240)],
        audio_features=_AudioFeatures(),
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=10,
    )
    assert on_beat.beat_sync > off_beat.beat_sync


def test_time_x_score_decreases_with_error() -> None:
    good = compute_audio_conditioned_score(
        [_obj("1", 0, 180)],
        audio_features=_AudioFeatures(),
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=10,
    )
    bad = compute_audio_conditioned_score(
        ["1,1,2,not-a-number,3,180"],
        audio_features=_AudioFeatures(),
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=10,
    )
    assert good.time_to_x_consistency >= bad.time_to_x_consistency


def test_object_budget_penalty_increases_when_over_budget() -> None:
    score = compute_audio_conditioned_score(
        [_obj("1", idx * 30, 180) for idx in range(10)],
        audio_features=_AudioFeatures(),
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=3,
    )
    assert score.object_budget_penalty > 0.0


def test_trigger_validity_score_decreases_with_orphans() -> None:
    valid = compute_audio_conditioned_score(
        [_group_obj("1", 10, 1), _trigger(20, 1)],
        audio_features=_AudioFeatures(),
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=10,
    )
    orphan = compute_audio_conditioned_score(
        [_trigger(20, 99)],
        audio_features=_AudioFeatures(),
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=10,
    )
    assert valid.trigger_validity > orphan.trigger_validity


def test_score_penalizes_editor_fatal_errors() -> None:
    clean = compute_audio_conditioned_score(
        [_obj("1", 0, 180)],
        audio_features=_AudioFeatures(),
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=10,
    )
    bad = compute_audio_conditioned_score(
        [_obj("1", 0, 180)],
        audio_features=_AudioFeatures(),
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=10,
        editor_issues=["fatal: unsupported_trigger"],
    )
    assert bad.editor_validity < clean.editor_validity


def test_score_penalizes_playability_warnings() -> None:
    clean = compute_audio_conditioned_score(
        [_obj("1", 0, 180), _obj("1", 300, 180)],
        audio_features=_AudioFeatures(),
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=10,
    )
    warned = compute_audio_conditioned_score(
        [_obj("1", 0, 180), _obj("1", 300, 180)],
        audio_features=_AudioFeatures(),
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=10,
        playability_warning_count=5,
    )
    assert warned.playability < clean.playability


def test_score_config_weights_are_applied() -> None:
    default = compute_audio_conditioned_score(
        [_obj("1", 0, 180)],
        audio_features=_AudioFeatures(),
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=10,
    )
    weighted = compute_audio_conditioned_score(
        [_obj("1", 0, 180)],
        audio_features=_AudioFeatures(),
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=10,
        scoring_config=ScoringConfig(weights={"beat_sync": 99.0}),
    )
    assert weighted.weights["beat_sync"] == 99.0
    assert weighted.total != default.total


def test_validation_report_serializes_to_dict() -> None:
    report = ValidationReport(score_breakdown={"total": 1.0}, editor_safety_report={"valid": True})
    payload = validation_report_to_dict(report)
    assert payload["score_breakdown"]["total"] == 1.0
    assert payload["editor_safety_report"]["valid"] is True


def test_score_total_is_deterministic() -> None:
    kwargs = {
        "audio_features": _AudioFeatures(),
        "speed_objects": [],
        "start_speed": "normal",
        "song_offset": 0.0,
        "beat_snap_tolerance": 0.1,
        "object_budget": 10,
    }
    score_a = compute_audio_conditioned_score([_obj("1", 0, 180)], **kwargs)
    score_b = compute_audio_conditioned_score([_obj("1", 0, 180)], **kwargs)
    assert score_a.to_dict() == score_b.to_dict()


def _sections_for_quality() -> list[SectionPlan]:
    return [
        SectionPlan(0, 1, 0, 200, "intro", "cube", SpeedState.NORMAL, 0.2, 0.2, 0.1, 0.2),
        SectionPlan(1, 2, 200, 400, "drop", "cube", SpeedState.NORMAL, 0.9, 0.8, 0.8, 0.8),
    ]


def test_score_penalizes_empty_drop_section() -> None:
    score = compute_audio_conditioned_score(
        [_obj("1", 20, 180)],
        audio_features=_AudioFeatures(),
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=10,
        section_plans=_sections_for_quality(),
    )

    assert score.empty_section_penalty > 0.0


def test_score_penalizes_excessive_repair_loss() -> None:
    score = compute_audio_conditioned_score(
        [_obj("1", 20, 180), _obj("36", 240, 180)],
        audio_features=_AudioFeatures(),
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=10,
        quality_metrics={"repair_loss_ratio": 0.6},
    )

    assert score.repair_loss_penalty == 0.6


def test_score_rewards_energy_density_alignment() -> None:
    aligned = compute_audio_conditioned_score(
        [_obj("1", 20, 180), _obj("36", 240, 180)],
        audio_features=_AudioFeatures(),
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=10,
        quality_metrics={"density_alignment_score": 0.9},
    )
    weak = compute_audio_conditioned_score(
        [_obj("1", 20, 180), _obj("36", 240, 180)],
        audio_features=_AudioFeatures(),
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=10,
        quality_metrics={"density_alignment_score": 0.2},
    )

    assert aligned.density_alignment > weak.density_alignment


def test_score_rewards_object_diversity() -> None:
    diverse = compute_audio_conditioned_score(
        [_obj("1", 20, 180), _obj("36", 240, 180), _obj("500", 300, 240)],
        audio_features=_AudioFeatures(),
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=10,
    )
    repetitive = compute_audio_conditioned_score(
        [_obj("1", 20, 180), _obj("1", 240, 180), _obj("1", 300, 180)],
        audio_features=_AudioFeatures(),
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=10,
    )

    assert diverse.object_diversity > repetitive.object_diversity


def test_score_rewards_reference_style_match() -> None:
    score = compute_audio_conditioned_score(
        [_obj("1", 20, 180), _obj("500", 240, 240)],
        audio_features=_AudioFeatures(),
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=10,
    )

    assert score.reference_style_match == score.style_consistency


def test_scoring_rewards_learned_style_match() -> None:
    score = compute_audio_conditioned_score(
        [_obj("1", 0, 180), _obj("500", 30, 240)],
        audio_features=_AudioFeatures(),
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=10,
        quality_metrics={
            "learned_style_match_score": 0.9,
            "learned_motif_match_score": 0.8,
            "learned_density_match_score": 0.7,
            "learned_trigger_usage_score": 0.6,
        },
    )

    assert score.learned_style_match == 0.9
    assert score.to_dict()["learned_motif_match"] == 0.8


def test_scoring_penalizes_deviation_from_learned_density() -> None:
    good = compute_audio_conditioned_score(
        [_obj("1", 0, 180), _obj("500", 30, 240)],
        audio_features=_AudioFeatures(),
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=10,
        quality_metrics={"learned_density_match_score": 0.9},
    )
    bad = compute_audio_conditioned_score(
        [_obj("1", 0, 180), _obj("500", 30, 240)],
        audio_features=_AudioFeatures(),
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=10,
        quality_metrics={"learned_density_match_score": 0.1},
    )

    assert good.learned_density_match > bad.learned_density_match
