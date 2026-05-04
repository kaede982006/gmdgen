from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from gmdgen.gd.plans import ObjectPlan, SectionPlan, TriggerPlan


@dataclass(slots=True)
class MusicEvent:
    time: float
    x: float
    event_type: str
    strength: float
    section_id: int
    priority: float
    suggested_gd_event_types: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RhythmPattern:
    pattern_id: str
    beat_indices: list[int]
    intervals: list[float]
    strengths: list[float]
    repetition_count: int
    suggested_motif_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AudioToGDAlignmentReport:
    strong_beat_coverage: float = 0.0
    downbeat_transition_coverage: float = 0.0
    onset_trigger_coverage: float = 0.0
    drop_transition_alignment: float = 0.0
    buildup_density_progression: float = 0.0
    sync_error_distribution: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_music_events(features: Any, beat_x_map: dict[int, float], section_plans: list[SectionPlan]) -> list[MusicEvent]:
    events: list[MusicEvent] = []
    beats = getattr(features, "beats", None) or getattr(features, "beat_features", [])
    for idx, beat in enumerate(beats):
        strength = float(getattr(beat, "strength", 0.0))
        is_downbeat = bool(getattr(beat, "is_downbeat", False))
        section_id = _section_id_for_time(float(getattr(beat, "time", 0.0)), section_plans)
        event_type = "downbeat" if is_downbeat else "strong_beat" if strength >= 0.65 else "weak_beat"
        suggestions = ["section_transition", "mode_change"] if is_downbeat else ["gameplay_event"] if strength >= 0.65 else ["decoration"]
        events.append(
            MusicEvent(
                time=float(getattr(beat, "time", 0.0)),
                x=float(beat_x_map.get(int(getattr(beat, "index", idx)), 0.0)),
                event_type=event_type,
                strength=strength,
                section_id=section_id,
                priority=1.0 if is_downbeat else strength,
                suggested_gd_event_types=suggestions,
            )
        )
    onsets = getattr(features, "onsets", [])
    for onset in onsets:
        strength = float(getattr(onset, "strength", 0.0))
        if strength < 0.5:
            continue
        time = float(getattr(onset, "time", 0.0))
        events.append(
            MusicEvent(
                time=time,
                x=_x_for_time(time, events),
                event_type="onset",
                strength=strength,
                section_id=_section_id_for_time(time, section_plans),
                priority=strength,
                suggested_gd_event_types=["pulse_trigger", "alpha_trigger", "visual_accent"],
            )
        )
    return sorted(events, key=lambda event: event.time)


def detect_repeated_rhythm(events: list[MusicEvent]) -> list[RhythmPattern]:
    strong = [event for event in events if event.event_type in {"strong_beat", "downbeat"}]
    patterns: list[RhythmPattern] = []
    window_size = 4 if len(strong) >= 4 else 3
    for idx in range(0, max(1, len(strong) - window_size + 1), window_size):
        window = strong[idx : idx + window_size]
        if len(window) < 3:
            continue
        intervals = [round(window[i + 1].time - window[i].time, 3) for i in range(len(window) - 1)]
        if len(set(intervals)) <= 2:
            patterns.append(
                RhythmPattern(
                    pattern_id=f"rhythm:{idx}",
                    beat_indices=list(range(idx, idx + len(window))),
                    intervals=intervals,
                    strengths=[event.strength for event in window],
                    repetition_count=1 + sum(1 for other in patterns if other.intervals == intervals),
                    suggested_motif_type="repeated_gameplay_motif",
                )
            )
    return patterns


def evaluate_audio_to_gd_alignment(
    music_events: list[MusicEvent],
    object_plans: list[ObjectPlan],
    trigger_plans: list[TriggerPlan],
    section_plans: list[SectionPlan],
) -> AudioToGDAlignmentReport:
    strong = [event for event in music_events if event.event_type in {"strong_beat", "downbeat"}]
    downbeats = [event for event in music_events if event.event_type == "downbeat"]
    onsets = [event for event in music_events if event.event_type == "onset"]
    strong_covered = sum(1 for event in strong if _near_object(event, object_plans, tolerance=48.0))
    downbeat_covered = sum(1 for event in downbeats if _near_section_boundary(event, section_plans))
    onset_covered = sum(1 for event in onsets if _near_trigger(event, trigger_plans, tolerance=64.0))
    drop_sections = [section for section in section_plans if section.section_type == "drop"]
    drop_alignment = 1.0 if not drop_sections else sum(1 for section in drop_sections if any(abs(event.time - section.start_time) <= 0.25 for event in strong)) / len(drop_sections)
    return AudioToGDAlignmentReport(
        strong_beat_coverage=strong_covered / max(1, len(strong)),
        downbeat_transition_coverage=downbeat_covered / max(1, len(downbeats)),
        onset_trigger_coverage=onset_covered / max(1, len(onsets)),
        drop_transition_alignment=drop_alignment,
        buildup_density_progression=_buildup_progression(section_plans, object_plans),
        sync_error_distribution={
            "object_avg": _avg_abs_sync_error([plan.sync_error for plan in object_plans]),
            "trigger_avg": _avg_abs_sync_error([plan.sync_error for plan in trigger_plans]),
        },
    )


def _section_id_for_time(time: float, section_plans: list[SectionPlan]) -> int:
    for idx, section in enumerate(section_plans):
        if section.start_time <= time <= section.end_time:
            return idx
    return max(0, len(section_plans) - 1)


def _x_for_time(time: float, events: list[MusicEvent]) -> float:
    if not events:
        return time * 240.0
    nearest = min(events, key=lambda event: abs(event.time - time))
    return nearest.x


def _near_object(event: MusicEvent, object_plans: list[ObjectPlan], *, tolerance: float) -> bool:
    return any(abs(plan.x - event.x) <= tolerance for plan in object_plans if plan.role not in {"safe_decoration", "ai_decoration"})


def _near_trigger(event: MusicEvent, trigger_plans: list[TriggerPlan], *, tolerance: float) -> bool:
    return any(abs(plan.x - event.x) <= tolerance for plan in trigger_plans)


def _near_section_boundary(event: MusicEvent, section_plans: list[SectionPlan]) -> bool:
    return any(abs(event.time - section.start_time) <= 0.25 or abs(event.time - section.end_time) <= 0.25 for section in section_plans)


def _buildup_progression(section_plans: list[SectionPlan], object_plans: list[ObjectPlan]) -> float:
    buildup = [(idx, section) for idx, section in enumerate(section_plans) if section.section_type == "buildup"]
    if len(buildup) < 2:
        return 1.0
    counts = []
    for idx, section in buildup:
        counts.append(sum(1 for plan in object_plans if section.start_x <= plan.x <= section.end_x))
    return 1.0 if all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1)) else 0.0


def _avg_abs_sync_error(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(abs(float(value)) for value in values) / len(values)
