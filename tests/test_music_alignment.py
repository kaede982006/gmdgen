# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass

from gmdgen.gd.plans import ObjectPlan, SectionPlan, TriggerPlan
from gmdgen.gd.time_mapping import SpeedState
from gmdgen.generate.music_alignment import build_music_events, detect_repeated_rhythm, evaluate_audio_to_gd_alignment


@dataclass
class Beat:
    time: float
    index: int
    strength: float
    is_downbeat: bool = False


@dataclass
class Onset:
    time: float
    strength: float


class Features:
    beats = [Beat(0, 0, 1.0, True), Beat(0.5, 1, 0.8), Beat(1.0, 2, 0.2), Beat(1.5, 3, 0.8)]
    beat_features = beats
    onsets = [Onset(0.0, 1.0), Onset(0.5, 0.9), Onset(1.0, 0.2)]


def _sections() -> list[SectionPlan]:
    return [
        SectionPlan(0, 1, 0, 240, "intro", "cube", SpeedState.NORMAL, 0.3, 0.2, 0.1, 0.3),
        SectionPlan(1, 2, 240, 520, "drop", "wave", SpeedState.NORMAL, 0.8, 0.8, 0.8, 0.7),
    ]


def test_strong_beats_get_gameplay_events() -> None:
    events = build_music_events(Features(), {0: 0, 1: 120, 2: 240, 3: 360}, _sections())

    strong = [event for event in events if event.event_type == "strong_beat"]
    assert strong
    assert "gameplay_event" in strong[0].suggested_gd_event_types


def test_weak_beats_do_not_force_gameplay_events() -> None:
    events = build_music_events(Features(), {0: 0, 1: 120, 2: 240, 3: 360}, _sections())

    weak = [event for event in events if event.event_type == "weak_beat"]
    assert weak
    assert weak[0].suggested_gd_event_types == ["decoration"]


def test_onsets_get_visual_trigger_candidates() -> None:
    events = build_music_events(Features(), {0: 0, 1: 120, 2: 240, 3: 360}, _sections())

    onset = [event for event in events if event.event_type == "onset"][0]
    assert "pulse_trigger" in onset.suggested_gd_event_types


def test_downbeats_align_section_transitions() -> None:
    events = build_music_events(Features(), {0: 0, 1: 120, 2: 240, 3: 360}, _sections())
    objects = [ObjectPlan("1", 0, 90, "ai_structure")]
    triggers = [TriggerPlan("pulse", "1006", 0, 300, target_group=1)]

    report = evaluate_audio_to_gd_alignment(events, objects, triggers, _sections())

    assert report.downbeat_transition_coverage > 0


def test_repeated_rhythm_creates_repeated_motif() -> None:
    events = build_music_events(Features(), {0: 0, 1: 120, 2: 240, 3: 360}, _sections())

    patterns = detect_repeated_rhythm(events)

    assert patterns
    assert patterns[0].suggested_motif_type == "repeated_gameplay_motif"
