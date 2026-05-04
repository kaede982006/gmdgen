# SPDX-License-Identifier: GPL-3.0-or-later
"""Multi-term scoring function for generated GD level objects.

Goodfellow §8.1 "How Learning Differs from Pure Optimization":
  Our current system does NOT use gradient descent. generation_passes is
  random restart search, not gradient update. However, having an explicit,
  well-defined objective function L_total improves candidate *selection*
  quality, even without backpropagation.

L_total =
    w_position    * L_position           (X-monotone ratio)
  + w_density     * L_density            (KL divergence from training density)
  + w_object_class* L_object_class       (style distribution match)
  + w_trigger     * L_trigger            (orphan trigger count)
  + w_section     * L_section            (section length distribution match)
  + w_richness    * L_richness           (original params preserved ratio)
  + w_diversity   * L_diversity          (unique object id ratio)
  + w_visible     * L_visible            (visible object ratio)

All terms return a *score* in [0, 1] where 1 is best.
L_total is a weighted sum; higher = better.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from gmdgen.features.tokenizer import (
    extract_object_id,
    extract_object_number,
)
from gmdgen.representation.object_classifier import (
    ObjectClass,
    classify,
    is_visible,
)

# ─────────────────────────────────────────────
# Score weights — tunable via config
# ─────────────────────────────────────────────

DEFAULT_WEIGHTS: dict[str, float] = {
    "position":     20.0,
    "density":      10.0,
    "object_class": 12.0,
    "trigger":      15.0,
    "section":       8.0,
    "richness":     15.0,
    "diversity":    10.0,
    "visible":      10.0,
}


@dataclass
class LevelScore:
    """Breakdown of each scoring term and the weighted total."""
    position: float = 0.0
    density: float = 0.0
    object_class: float = 0.0
    trigger: float = 0.0
    section: float = 0.0
    richness: float = 0.0
    diversity: float = 0.0
    visible: float = 0.0
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    @property
    def total(self) -> float:
        return (
            self.weights.get("position", 20.0)     * self.position
            + self.weights.get("density", 10.0)    * self.density
            + self.weights.get("object_class", 12.0) * self.object_class
            + self.weights.get("trigger", 15.0)    * self.trigger
            + self.weights.get("section", 8.0)     * self.section
            + self.weights.get("richness", 15.0)   * self.richness
            + self.weights.get("diversity", 10.0)  * self.diversity
            + self.weights.get("visible", 10.0)    * self.visible
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "total": round(self.total, 4),
            "position": round(self.position, 4),
            "density": round(self.density, 4),
            "object_class": round(self.object_class, 4),
            "trigger": round(self.trigger, 4),
            "section": round(self.section, 4),
            "richness": round(self.richness, 4),
            "diversity": round(self.diversity, 4),
            "visible": round(self.visible, 4),
        }


# ─────────────────────────────────────────────
# Individual scoring functions — each returns [0, 1]
# ─────────────────────────────────────────────

def score_position(objects: list[str]) -> float:
    """L_position: fraction of consecutive pairs where x[i] >= x[i-1].

    Goodfellow Ch.4 §4.4: a hard constraint relaxed into a soft penalty.
    Perfect monotone layout → 1.0.
    """
    prev: float | None = None
    ordered = 0
    total = 0
    for obj in objects:
        x = extract_object_number(obj, "2")
        if x is None:
            continue
        if prev is not None:
            total += 1
            if x >= prev:
                ordered += 1
        prev = x
    return ordered / total if total > 0 else 1.0


def score_density(
    objects: list[str],
    *,
    reference_density: list[float] | None = None,
    grid_unit: int = 30,
) -> float:
    """L_density: smoothness of per-grid object density distribution.

    Without a reference, measures uniformity via normalised entropy.
    With a reference distribution, returns 1 - Jensen-Shannon divergence.
    Goodfellow Ch.3 §3.13 "Information Theory".
    """
    bucket_counts: Counter[int] = Counter()
    for obj in objects:
        x = extract_object_number(obj, "2")
        if x is not None:
            bucket_counts[int(x) // grid_unit] += 1

    if not bucket_counts:
        return 0.0

    total = sum(bucket_counts.values())
    if total == 0:
        return 0.0

    if reference_density:
        # Jensen-Shannon divergence between generated and reference
        n = max(len(reference_density), max(bucket_counts) + 1)
        ref = [reference_density[i] if i < len(reference_density) else 0.0 for i in range(n)]
        ref_total = sum(ref)
        if ref_total <= 0:
            ref = [1.0 / n] * n
            ref_total = 1.0

        gen = [bucket_counts.get(i, 0) / total for i in range(n)]
        ref_norm = [r / ref_total for r in ref]

        def _kl(p: list[float], q: list[float]) -> float:
            eps = 1e-10
            return sum(
                pi * math.log((pi + eps) / (qi + eps))
                for pi, qi in zip(p, q)
                if pi > 0
            )

        m = [(pi + qi) / 2 for pi, qi in zip(gen, ref_norm)]
        jsd = 0.5 * _kl(gen, m) + 0.5 * _kl(ref_norm, m)
        return max(0.0, 1.0 - min(1.0, jsd))

    # No reference: entropy-based smoothness
    probs = [c / total for c in bucket_counts.values()]
    max_entropy = math.log2(len(probs)) if len(probs) > 1 else 1.0
    if max_entropy <= 0:
        return 1.0
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return min(1.0, entropy / max_entropy)


def score_object_class(
    objects: list[str],
    *,
    target_distribution: dict[str, float] | None = None,
) -> float:
    """L_object_class: match between class distribution and reference.

    Goodfellow Ch.1: factors of variation — object class is one such factor.
    Returns 1.0 if no reference is given (no penalty).
    """
    if not target_distribution:
        return 1.0

    if not objects:
        return 0.0

    counts: Counter[str] = Counter()
    for obj in objects:
        obj_id = extract_object_id(obj)
        if obj_id:
            counts[classify(obj_id).value] += 1

    total = sum(counts.values()) or 1

    # sum of |actual_ratio - target_ratio| per class, normalised to [0,1]
    diff = 0.0
    for cls_name, target_ratio in target_distribution.items():
        actual = counts.get(cls_name, 0) / total
        diff += abs(actual - target_ratio)

    return max(0.0, 1.0 - min(1.0, diff))


def score_trigger_integrity(objects: list[str]) -> float:
    """L_trigger: fraction of triggers whose target group exists in level.

    Goodfellow Ch.16 §16.2 "Graphs to Describe Model Structure":
    Trigger→group is an edge in the relational graph. Missing edges = penalty.
    """
    from gmdgen.generate.repairer import _TRIGGER_IDS, _GROUP_KEY, _TRIGGER_TARGET_KEY
    from gmdgen.features.tokenizer import extract_object_field

    defined_groups: set[int] = set()
    for obj in objects:
        raw = extract_object_field(obj, _GROUP_KEY)
        if raw:
            for part in raw.split("."):
                p = part.strip()
                if p.isdigit():
                    defined_groups.add(int(p))

    valid_triggers = 0
    orphan_triggers = 0
    for obj in objects:
        obj_id = extract_object_id(obj)
        if obj_id not in _TRIGGER_IDS:
            continue
        raw_target = extract_object_field(obj, _TRIGGER_TARGET_KEY)
        if raw_target is None:
            continue
        raw_target = raw_target.strip()
        if not raw_target.isdigit():
            continue
        if int(raw_target) in defined_groups:
            valid_triggers += 1
        else:
            orphan_triggers += 1

    total_triggers = valid_triggers + orphan_triggers
    if total_triggers == 0:
        return 1.0
    return valid_triggers / total_triggers


def score_section_length(
    objects: list[str],
    *,
    reference_section_lengths: list[int] | None = None,
) -> float:
    """L_section: section length distribution vs reference.

    Without reference, measures variance smoothness.
    With reference, returns correlation with expected section length distribution.
    """
    from gmdgen.data.preprocess import detect_section_boundaries

    boundaries = detect_section_boundaries(objects)
    if len(boundaries) < 2:
        return 0.5

    lengths: list[int] = []
    sorted_boundaries = sorted(boundaries, key=lambda b: b.start_object_index)
    for i, boundary in enumerate(sorted_boundaries):
        next_start = (
            sorted_boundaries[i + 1].start_object_index
            if i + 1 < len(sorted_boundaries)
            else len(objects)
        )
        lengths.append(next_start - boundary.start_object_index)

    if not lengths:
        return 0.5

    if not reference_section_lengths:
        # Measure coefficient of variation (lower = more uniform)
        mean_len = sum(lengths) / len(lengths)
        if mean_len <= 0:
            return 0.5
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        cv = math.sqrt(variance) / mean_len
        return max(0.0, 1.0 - min(1.0, cv / 2.0))

    # Pearson correlation with reference lengths (binned to same scale)
    n = min(len(lengths), len(reference_section_lengths))
    if n < 2:
        return 0.5
    gen = lengths[:n]
    ref = reference_section_lengths[:n]
    mean_g = sum(gen) / n
    mean_r = sum(ref) / n
    cov = sum((g - mean_g) * (r - mean_r) for g, r in zip(gen, ref)) / n
    std_g = math.sqrt(sum((g - mean_g) ** 2 for g in gen) / n)
    std_r = math.sqrt(sum((r - mean_r) ** 2 for r in ref) / n)
    if std_g < 1e-9 or std_r < 1e-9:
        return 0.5
    corr = cov / (std_g * std_r)
    return (corr + 1.0) / 2.0  # map [-1,1] → [0,1]


def score_richness(objects: list[str]) -> float:
    """L_richness: fraction of objects retaining original GD parameters.

    Objects with > 6 comma-separated fields have non-trivial parameters.
    Plain 1,id,2,x,3,y = 6 fields → "simplified".
    Goodfellow Ch.7: regularisation — safe_simplify = capacity reduction.
    """
    if not objects:
        return 0.0
    rich = sum(1 for obj in objects if len(obj.split(",")) > 6)
    return rich / len(objects)


def score_diversity(objects: list[str]) -> float:
    """L_diversity: ratio of unique object IDs to total objects."""
    if not objects:
        return 0.0
    ids = [extract_object_id(obj) for obj in objects]
    valid_ids = [i for i in ids if i]
    if not valid_ids:
        return 0.0
    return len(set(valid_ids)) / len(valid_ids)


def score_visible(objects: list[str]) -> float:
    """L_visible: fraction of objects with a visible class."""
    if not objects:
        return 0.0
    vis = sum(
        1 for obj in objects
        if (obj_id := extract_object_id(obj)) and is_visible(obj_id)
    )
    return vis / len(objects)


# ─────────────────────────────────────────────
# Composite scorer
# ─────────────────────────────────────────────

def compute_level_score(
    objects: list[str],
    *,
    weights: dict[str, float] | None = None,
    target_class_distribution: dict[str, float] | None = None,
    reference_density: list[float] | None = None,
    reference_section_lengths: list[int] | None = None,
    grid_unit: int = 30,
) -> LevelScore:
    """Compute all scoring terms and return a LevelScore dataclass."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    score = LevelScore(weights=w)
    score.position = score_position(objects)
    score.density = score_density(
        objects, reference_density=reference_density, grid_unit=grid_unit
    )
    score.object_class = score_object_class(
        objects, target_distribution=target_class_distribution
    )
    score.trigger = score_trigger_integrity(objects)
    score.section = score_section_length(
        objects, reference_section_lengths=reference_section_lengths
    )
    score.richness = score_richness(objects)
    score.diversity = score_diversity(objects)
    score.visible = score_visible(objects)
    return score


@dataclass
class ScoringConfig:
    weights: dict[str, float] = field(default_factory=dict)
    editor_fatal_penalty: float = 0.25
    playability_warning_penalty: float = 0.03


@dataclass
class AudioConditionedScore:
    """Scoring terms for audio-conditioned structured generation.

    These are search objectives, not gradient-descent losses. They mirror the
    requested L_total terms while keeping the existing generator deterministic
    and dependency-light.
    """

    beat_sync: float = 0.0
    onset_sync: float = 0.0
    section_sync: float = 0.0
    time_to_x_consistency: float = 0.0
    speed_portal_consistency: float = 0.0
    energy_density: float = 0.0
    style_consistency: float = 0.0
    playability: float = 0.0
    trigger_validity: float = 0.0
    group_validity: float = 0.0
    editor_validity: float = 0.0
    object_budget: float = 0.0
    object_budget_penalty: float = 0.0
    overcrowding_penalty: float = 0.0
    motif_quality: float = 0.0
    section_contrast: float = 0.0
    drop_impact: float = 0.0
    density_alignment: float = 0.0
    object_diversity: float = 0.0
    trigger_usefulness: float = 0.0
    reference_style_match: float = 0.0
    learned_style_match: float = 0.0
    learned_motif_match: float = 0.0
    learned_density_match: float = 0.0
    learned_trigger_usage: float = 0.0
    repair_loss_penalty: float = 0.0
    empty_section_penalty: float = 0.0
    repetitive_pattern_penalty: float = 0.0
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "beat_sync": 18.0,
            "onset_sync": 10.0,
            "section_sync": 8.0,
            "time_to_x_consistency": 18.0,
            "speed_portal_consistency": 12.0,
            "energy_density": 8.0,
            "style_consistency": 6.0,
            "playability": 12.0,
            "trigger_validity": 14.0,
            "group_validity": 12.0,
            "editor_validity": 10.0,
            "object_budget": 8.0,
            "object_budget_penalty": 8.0,
            "overcrowding_penalty": 8.0,
            "motif_quality": 7.0,
            "section_contrast": 7.0,
            "drop_impact": 10.0,
            "density_alignment": 8.0,
            "object_diversity": 6.0,
            "trigger_usefulness": 6.0,
            "reference_style_match": 5.0,
            "learned_style_match": 6.0,
            "learned_motif_match": 5.0,
            "learned_density_match": 5.0,
            "learned_trigger_usage": 4.0,
            "repair_loss_penalty": 10.0,
            "empty_section_penalty": 12.0,
            "repetitive_pattern_penalty": 6.0,
        }
    )

    @property
    def total(self) -> float:
        positive_terms = [
            "beat_sync",
            "onset_sync",
            "section_sync",
            "time_to_x_consistency",
            "speed_portal_consistency",
            "energy_density",
            "style_consistency",
            "playability",
            "trigger_validity",
            "group_validity",
            "editor_validity",
            "object_budget",
            "motif_quality",
            "section_contrast",
            "drop_impact",
            "density_alignment",
            "object_diversity",
            "trigger_usefulness",
            "reference_style_match",
            "learned_style_match",
            "learned_motif_match",
            "learned_density_match",
            "learned_trigger_usage",
        ]
        total = sum(self.weights.get(name, 0.0) * getattr(self, name) for name in positive_terms)
        total -= self.weights.get("object_budget_penalty", 0.0) * self.object_budget_penalty
        total -= self.weights.get("overcrowding_penalty", 0.0) * self.overcrowding_penalty
        total -= self.weights.get("repair_loss_penalty", 0.0) * self.repair_loss_penalty
        total -= self.weights.get("empty_section_penalty", 0.0) * self.empty_section_penalty
        total -= self.weights.get("repetitive_pattern_penalty", 0.0) * self.repetitive_pattern_penalty
        return max(0.0, total)

    def to_dict(self) -> dict[str, float]:
        return {
            "total": round(self.total, 4),
            "beat_sync": round(self.beat_sync, 4),
            "onset_sync": round(self.onset_sync, 4),
            "section_sync": round(self.section_sync, 4),
            "section_alignment": round(self.section_sync, 4),
            "time_to_x_consistency": round(self.time_to_x_consistency, 4),
            "speed_portal_consistency": round(self.speed_portal_consistency, 4),
            "energy_density": round(self.energy_density, 4),
            "style_consistency": round(self.style_consistency, 4),
            "playability": round(self.playability, 4),
            "playability_safety": round(self.playability, 4),
            "trigger_validity": round(self.trigger_validity, 4),
            "group_validity": round(self.group_validity, 4),
            "editor_validity": round(self.editor_validity, 4),
            "object_budget": round(self.object_budget, 4),
            "object_budget_penalty": round(self.object_budget_penalty, 4),
            "overcrowding_penalty": round(self.overcrowding_penalty, 4),
            "motif_quality": round(self.motif_quality, 4),
            "section_contrast": round(self.section_contrast, 4),
            "drop_impact": round(self.drop_impact, 4),
            "density_alignment": round(self.density_alignment, 4),
            "object_diversity": round(self.object_diversity, 4),
            "trigger_usefulness": round(self.trigger_usefulness, 4),
            "reference_style_match": round(self.reference_style_match, 4),
            "learned_style_match": round(self.learned_style_match, 4),
            "learned_motif_match": round(self.learned_motif_match, 4),
            "learned_density_match": round(self.learned_density_match, 4),
            "learned_trigger_usage": round(self.learned_trigger_usage, 4),
            "repair_loss_penalty": round(self.repair_loss_penalty, 4),
            "empty_section_penalty": round(self.empty_section_penalty, 4),
            "repetitive_pattern_penalty": round(self.repetitive_pattern_penalty, 4),
        }


ScoreBreakdown = AudioConditionedScore


def compute_audio_conditioned_score(
    objects: list[str],
    *,
    audio_features: Any,
    speed_objects: list[Any],
    start_speed: Any,
    song_offset: float,
    beat_snap_tolerance: float,
    object_budget: int,
    editor_issues: list[str] | None = None,
    playability_warning_count: int = 0,
    scoring_config: ScoringConfig | None = None,
    section_plans: list[Any] | None = None,
    quality_metrics: dict[str, Any] | None = None,
) -> AudioConditionedScore:
    from gmdgen.features.tokenizer import extract_object_field
    from gmdgen.gd.time_mapping import time_for_pos_like_gd
    from gmdgen.representation.object_classifier import ObjectClass, classify

    score = AudioConditionedScore()
    config = scoring_config or ScoringConfig()
    if config.weights:
        score.weights.update(config.weights)
    beat_times: list[float] = list(getattr(audio_features, "beat_times", []) or [])
    onset_times: list[float] = list(getattr(audio_features, "onset_times", []) or [])
    sections = list(getattr(audio_features, "sections", []) or [])

    event_times: list[float] = []
    gameplay_event_times: list[float] = []
    trigger_times: list[float] = []
    x_values: list[float] = []
    for obj in objects:
        obj_id = extract_object_id(obj)
        x = extract_object_number(obj, "2")
        if obj_id is None or x is None:
            continue
        event_time = time_for_pos_like_gd(
            x,
            speed_objects,
            start_speed=start_speed,
            song_offset=song_offset,
        )
        event_times.append(event_time)
        x_values.append(x)
        obj_class = classify(obj_id)
        if obj_class == ObjectClass.TRIGGER:
            trigger_times.append(event_time)
        elif obj_class in {ObjectClass.STRUCTURE, ObjectClass.SPECIAL} or obj_id in {"8", "39"}:
            gameplay_event_times.append(event_time)

    score.beat_sync = _alignment_score(
        gameplay_event_times or event_times,
        beat_times,
        beat_snap_tolerance,
    )
    score.onset_sync = _alignment_score(
        trigger_times,
        onset_times or beat_times,
        max(beat_snap_tolerance, 0.12),
    )
    score.section_sync = _section_boundary_score(event_times, sections)
    score.time_to_x_consistency = _roundtrip_consistency(
        objects,
        speed_objects=speed_objects,
        start_speed=start_speed,
        song_offset=song_offset,
        tolerance=beat_snap_tolerance,
    )
    score.speed_portal_consistency = _speed_portal_consistency(
        speed_objects,
        start_speed=start_speed,
        song_offset=song_offset,
        tolerance=beat_snap_tolerance,
    )
    score.energy_density = _energy_density_score(objects, sections)
    score.style_consistency = compute_level_score(objects).object_class
    score.playability = max(
        0.0,
        _playability_spacing_score(x_values)
        - min(0.5, playability_warning_count * config.playability_warning_penalty),
    )
    score.trigger_validity = score_trigger_integrity(objects)
    score.group_validity = _group_validity_score(objects)
    fatal_count = sum(1 for issue in (editor_issues or []) if "fatal" in issue or "unsupported" in issue or "nan" in issue)
    editor_issue_penalty = min(
        0.8,
        len(editor_issues or []) * 0.08 + fatal_count * config.editor_fatal_penalty,
    )
    score.editor_validity = max(0.0, min(
        score_position(objects),
        score.trigger_validity,
        score.group_validity,
        1.0 if len(objects) <= max(object_budget, 1) else 0.4,
    ) - editor_issue_penalty)
    score.object_budget = 1.0 if len(objects) <= max(object_budget, 1) else max(0.0, object_budget / len(objects))
    score.object_budget_penalty = max(0.0, (len(objects) - max(object_budget, 1)) / max(len(objects), 1))
    score.overcrowding_penalty = _overcrowding_penalty(objects)
    quality_metrics = quality_metrics or {}
    score.density_alignment = _clamp01(float(quality_metrics.get("density_alignment_score", score.energy_density)))
    score.drop_impact = _clamp01(float(quality_metrics.get("drop_impact_score", _drop_impact_from_objects(objects, section_plans or []))))
    score.section_contrast = _section_contrast_score(section_plans or [], objects)
    score.object_diversity = score_diversity(objects)
    score.trigger_usefulness = _trigger_usefulness_score(trigger_times, onset_times or beat_times)
    score.reference_style_match = score.style_consistency
    score.learned_style_match = _clamp01(float(quality_metrics.get("learned_style_match_score", score.reference_style_match)))
    score.learned_motif_match = _clamp01(float(quality_metrics.get("learned_motif_match_score", 0.0)))
    score.learned_density_match = _clamp01(float(quality_metrics.get("learned_density_match_score", score.density_alignment)))
    score.learned_trigger_usage = _clamp01(float(quality_metrics.get("learned_trigger_usage_score", score.trigger_usefulness)))
    score.motif_quality = _motif_quality_score(objects)
    score.repair_loss_penalty = _clamp01(float(quality_metrics.get("repair_loss_ratio", 0.0)))
    score.empty_section_penalty = _empty_section_penalty(section_plans or [], objects)
    score.repetitive_pattern_penalty = _repetitive_pattern_penalty(objects)
    return score


def score_breakdown_to_dict(score: AudioConditionedScore) -> dict[str, float]:
    return score.to_dict()


def validation_report_to_dict(report: Any) -> dict[str, Any]:
    if hasattr(report, "to_dict"):
        return report.to_dict()
    return dict(report)


def _alignment_score(
    event_times: list[float],
    target_times: list[float],
    tolerance: float,
) -> float:
    if not event_times or not target_times:
        return 1.0
    
    import bisect
    sorted_targets = sorted(target_times)
    tolerance = max(tolerance, 1e-6)
    values: list[float] = []
    
    for event_time in event_times:
        idx = bisect.bisect_left(sorted_targets, event_time)
        if idx == 0:
            nearest = sorted_targets[0]
        elif idx == len(sorted_targets):
            nearest = sorted_targets[-1]
        else:
            prev = sorted_targets[idx - 1]
            curr = sorted_targets[idx]
            nearest = prev if event_time - prev < curr - event_time else curr
            
        error = abs(event_time - nearest)
        values.append(math.exp(-error / tolerance))
        
    return sum(values) / len(values)


def _section_boundary_score(event_times: list[float], sections: list[Any]) -> float:
    if not sections:
        return 1.0
    boundaries = [float(getattr(section, "start_time", 0.0)) for section in sections[1:]]
    if not boundaries:
        return 1.0
    return _alignment_score(event_times, boundaries, 0.5)


def _roundtrip_consistency(
    objects: list[str],
    *,
    speed_objects: list[Any],
    start_speed: Any,
    song_offset: float,
    tolerance: float,
) -> float:
    from gmdgen.gd.time_mapping import pos_for_time_like_gd, time_for_pos_like_gd

    if not objects:
        return 1.0
    values: list[float] = []
    for obj in objects:
        x = extract_object_number(obj, "2")
        if x is None:
            continue
        time_value = time_for_pos_like_gd(
            x,
            speed_objects,
            start_speed=start_speed,
            song_offset=song_offset,
        )
        roundtrip_x = pos_for_time_like_gd(
            time_value,
            speed_objects,
            start_speed=start_speed,
            song_offset=song_offset,
        )
        values.append(math.exp(-abs(roundtrip_x - x) / max(1.0, tolerance * 300.0)))
    return sum(values) / len(values) if values else 1.0


def _speed_portal_consistency(
    speed_objects: list[Any],
    *,
    start_speed: Any,
    song_offset: float,
    tolerance: float,
) -> float:
    from gmdgen.gd.time_mapping import time_for_pos_like_gd

    if not speed_objects:
        return 1.0
    sorted_by_x = sorted(speed_objects, key=lambda obj: obj.x)
    if list(speed_objects) != sorted_by_x:
        return 0.0
    values: list[float] = []
    for speed_object in speed_objects:
        actual = time_for_pos_like_gd(
            speed_object.x,
            speed_objects,
            start_speed=start_speed,
            song_offset=song_offset,
        )
        values.append(math.exp(-abs(actual - speed_object.time) / max(tolerance, 1e-6)))
    return sum(values) / len(values)


def _energy_density_score(objects: list[str], sections: list[Any]) -> float:
    if not objects or not sections:
        return 1.0
    x_values = [extract_object_number(obj, "2") for obj in objects]
    x_values = [x for x in x_values if x is not None]
    if not x_values:
        return 0.0
    total_width = max(x_values) - min(x_values)
    if total_width <= 0:
        return 0.5

    # Use temporal section energy only as a monotonic target. Exact x boundaries
    # are handled by audio_conditioned.SectionPlan during generation.
    energies = [float(getattr(section, "mean_energy", 0.0)) for section in sections]
    if len(energies) < 2 or max(energies) <= 0:
        return 1.0
    expected_rank = sorted(range(len(energies)), key=lambda idx: energies[idx])
    # Approximate generated density by object order split into equal bins.
    sorted_x = sorted(x_values)
    bin_size = max(1, len(sorted_x) // len(energies))
    densities = []
    for idx in range(len(energies)):
        chunk = sorted_x[idx * bin_size : (idx + 1) * bin_size]
        if len(chunk) < 2:
            densities.append(0.0)
        else:
            densities.append(len(chunk) / max(1.0, chunk[-1] - chunk[0]))
    density_rank = sorted(range(len(densities)), key=lambda idx: densities[idx])
    matches = sum(1 for a, b in zip(expected_rank, density_rank) if a == b)
    return matches / len(energies)


def _playability_spacing_score(x_values: list[float]) -> float:
    if len(x_values) < 2:
        return 1.0
    xs = sorted(x_values)
    gaps = [b - a for a, b in zip(xs, xs[1:]) if b >= a]
    if not gaps:
        return 0.0
    too_tight = sum(1 for gap in gaps if gap < 8)
    too_wide = sum(1 for gap in gaps if gap > 900)
    penalty = (too_tight + too_wide * 0.5) / len(gaps)
    return max(0.0, 1.0 - penalty)


def _overcrowding_penalty(objects: list[str], *, grid_unit: int = 30, max_per_grid: int = 8) -> float:
    if not objects:
        return 0.0
    buckets: Counter[int] = Counter()
    for obj in objects:
        x_value = extract_object_number(obj, "2")
        if x_value is not None:
            buckets[int(x_value) // grid_unit] += 1
    if not buckets:
        return 0.0
    overflow = sum(max(0, count - max_per_grid) for count in buckets.values())
    return min(1.0, overflow / max(1, len(objects)))


def _group_validity_score(objects: list[str]) -> float:
    from gmdgen.features.tokenizer import extract_object_field

    group_counts = 0
    invalid = 0
    for obj in objects:
        raw = extract_object_field(obj, "155")
        if not raw:
            continue
        for part in raw.split("."):
            group_counts += 1
            if not part.strip().isdigit() or int(part.strip()) <= 0:
                invalid += 1
    if group_counts == 0:
        return 1.0
    return 1.0 - invalid / group_counts


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _trigger_usefulness_score(trigger_times: list[float], onset_times: list[float]) -> float:
    if not trigger_times:
        return 0.4
    return _alignment_score(trigger_times, onset_times, 0.16) if onset_times else min(1.0, len(trigger_times) / 8.0)


def _motif_quality_score(objects: list[str]) -> float:
    ids = [extract_object_id(obj) for obj in objects if extract_object_id(obj)]
    if len(ids) < 4:
        return 0.25 if ids else 0.0
    windows = [tuple(ids[idx:idx + 3]) for idx in range(0, max(0, len(ids) - 2))]
    repeated = len(windows) - len(set(windows))
    variation = len(set(ids)) / len(ids)
    return _clamp01(0.45 * min(1.0, repeated / max(1, len(windows) / 4)) + 0.55 * variation)


def _drop_impact_from_objects(objects: list[str], section_plans: list[Any]) -> float:
    drops = [section for section in section_plans if getattr(section, "section_type", "") == "drop"]
    if not drops:
        return 1.0
    total_density = max(1, len(objects))
    values = []
    for section in drops:
        start = float(getattr(section, "start_x", 0.0))
        end = float(getattr(section, "end_x", start + 1.0))
        count = 0
        triggers = 0
        for obj in objects:
            x = extract_object_number(obj, "2")
            if x is None or not (start <= x <= end):
                continue
            count += 1
            cls = classify(extract_object_id(obj) or "")
            if cls == ObjectClass.TRIGGER:
                triggers += 1
        width = max(1.0, end - start)
        values.append(min(1.0, (count / width * 1000.0) / max(1.0, total_density / 10.0) + min(0.25, triggers * 0.04)))
    return sum(values) / len(values)


def _section_contrast_score(section_plans: list[Any], objects: list[str]) -> float:
    if len(section_plans) < 2:
        return 1.0
    densities = []
    for section in section_plans:
        start = float(getattr(section, "start_x", 0.0))
        end = float(getattr(section, "end_x", start + 1.0))
        count = sum(
            1 for obj in objects
            if (x := extract_object_number(obj, "2")) is not None and start <= x <= end
        )
        densities.append(count / max(1.0, end - start))
    if not densities:
        return 0.0
    return _clamp01((max(densities) - min(densities)) / max(max(densities), 1e-9))


def _empty_section_penalty(section_plans: list[Any], objects: list[str]) -> float:
    if not section_plans:
        return 0.0
    empty = 0
    heavy_empty = 0
    for section in section_plans:
        start = float(getattr(section, "start_x", 0.0))
        end = float(getattr(section, "end_x", start + 1.0))
        count = sum(
            1 for obj in objects
            if (x := extract_object_number(obj, "2")) is not None and start <= x <= end
        )
        if count == 0:
            empty += 1
            if getattr(section, "section_type", "") == "drop":
                heavy_empty += 1
    return _clamp01((empty + heavy_empty * 2) / max(1, len(section_plans)))


def _repetitive_pattern_penalty(objects: list[str]) -> float:
    ids = [extract_object_id(obj) for obj in objects if extract_object_id(obj)]
    if len(ids) < 4:
        return 0.0
    most_common = Counter(ids).most_common(1)[0][1]
    return _clamp01((most_common / len(ids) - 0.45) * 1.8)


def build_reference_stats_from_records(
    records: list,
    *,
    grid_unit: int = 30,
    max_density_buckets: int = 500,
) -> dict[str, Any]:
    """Pre-compute reference stats from training records for use in scoring.

    Call once after training and cache in the artifact.
    """
    from gmdgen.data.preprocess import (
        detect_section_boundaries,
        split_level_objects,
    )
    from gmdgen.representation.object_classifier import classify

    all_density: Counter[int] = Counter()
    all_class_counts: Counter[str] = Counter()
    all_section_lengths: list[int] = []
    total_objs = 0

    for record in records:
        objects = split_level_objects(record.decoded_level_data)
        for obj in objects:
            x = extract_object_number(obj, "2")
            if x is not None:
                bucket = int(x) // grid_unit
                if bucket < max_density_buckets:
                    all_density[bucket] += 1

            obj_id = extract_object_id(obj)
            if obj_id:
                all_class_counts[classify(obj_id).value] += 1
                total_objs += 1

        boundaries = detect_section_boundaries(objects)
        sorted_bounds = sorted(boundaries, key=lambda b: b.start_object_index)
        for i, b in enumerate(sorted_bounds):
            end = (
                sorted_bounds[i + 1].start_object_index
                if i + 1 < len(sorted_bounds)
                else len(objects)
            )
            all_section_lengths.append(end - b.start_object_index)

    density_buckets = max_density_buckets
    density_distribution = [
        all_density.get(b, 0) / max(1, sum(all_density.values()))
        for b in range(density_buckets)
    ]

    class_distribution = {
        cls: round(cnt / max(1, total_objs), 6)
        for cls, cnt in all_class_counts.items()
    }

    return {
        "density_distribution": density_distribution,
        "class_distribution": class_distribution,
        "section_lengths": all_section_lengths[:2000],
    }
