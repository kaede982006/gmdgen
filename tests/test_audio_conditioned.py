from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from gmdgen.audio.analysis import AudioConfidenceReport, analyze_audio
from gmdgen.generate.audio_conditioned import GroupAllocator, _plan_sections
from gmdgen.generate.generator import generate_from_config
from gmdgen.generate.scoring import compute_audio_conditioned_score
from gmdgen.gd.time_mapping import build_beat_x_map, round_trip_error_report
from gmdgen.io.gmd_parser import parse_gmd_file


def _write_click_wav(path: Path, *, duration: float = 3.0, bpm: float = 120.0) -> None:
    sample_rate = 8000
    samples = [0.0] * int(sample_rate * duration)
    interval = 60.0 / bpm
    click_len = int(0.025 * sample_rate)
    t = 0.0
    while t < duration:
        start = int(t * sample_rate)
        for idx in range(start, min(len(samples), start + click_len)):
            phase = (idx - start) / max(1, click_len)
            samples[idx] += math.sin(phase * math.pi) * 0.9
        t += interval

    pcm = b"".join(
        struct.pack("<h", max(-32768, min(32767, int(sample * 32767))))
        for sample in samples
    )
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm)


def test_analyze_audio_extracts_basic_features(tmp_path: Path) -> None:
    audio_path = tmp_path / "clicks.wav"
    _write_click_wav(audio_path)

    features = analyze_audio(audio_path, target_duration=3.0)

    assert features.duration == 3.0
    assert features.bpm > 0
    assert features.beat_times
    assert features.frame_features
    assert features.mel_spectrogram
    assert features.sections


def test_audio_analysis_fallback_outputs_required_fields(tmp_path: Path) -> None:
    audio_path = tmp_path / "clicks.wav"
    _write_click_wav(audio_path)

    features = analyze_audio(audio_path, target_duration=3.0, backend="fallback")

    assert features.backend == "fallback_wav"
    assert features.sample_rate > 0
    assert features.beats
    assert all(hasattr(beat, "local_energy") for beat in features.beats)
    assert features.onsets
    assert all(hasattr(onset, "strength") for onset in features.onsets)
    assert features.rms_envelope
    assert features.fallback_flux
    assert features.sections
    assert all(section.end_time >= section.start_time for section in features.sections)
    assert 0.0 <= features.confidence <= 1.0
    assert features.confidence_report is not None


def test_onset_strength_normalized(tmp_path: Path) -> None:
    audio_path = tmp_path / "clicks.wav"
    _write_click_wav(audio_path)

    features = analyze_audio(audio_path, target_duration=3.0, backend="fallback")

    assert all(0.0 <= onset.strength <= 1.0 for onset in features.onsets)
    assert all(0.0 <= value <= 1.0 for _, value in features.onset_envelope)


def test_rms_smoothing_non_empty(tmp_path: Path) -> None:
    audio_path = tmp_path / "clicks.wav"
    _write_click_wav(audio_path)

    features = analyze_audio(audio_path, target_duration=3.0, backend="fallback")

    assert features.rms_envelope
    assert all(value >= 0.0 for _, value in features.rms_envelope)


def test_audio_confidence_report_range(tmp_path: Path) -> None:
    audio_path = tmp_path / "clicks.wav"
    _write_click_wav(audio_path)

    features = analyze_audio(audio_path, target_duration=3.0, backend="fallback")
    report = features.confidence_report

    assert report is not None
    for value in [
        report.overall,
        report.bpm_confidence,
        report.beat_confidence,
        report.onset_confidence,
        report.section_confidence,
        report.tempo_stability,
    ]:
        assert 0.0 <= value <= 1.0


def test_low_confidence_reduces_trigger_intensity(tmp_path: Path) -> None:
    audio_path = tmp_path / "clicks.wav"
    _write_click_wav(audio_path, duration=3.0)
    features = analyze_audio(audio_path, target_duration=3.0, backend="fallback")
    assert features.confidence_report is not None

    features.confidence_report = AudioConfidenceReport(
        overall=1.0,
        bpm_confidence=1.0,
        beat_confidence=1.0,
        onset_confidence=1.0,
        section_confidence=1.0,
        tempo_stability=1.0,
        backend="fallback_wav",
    )
    high = _plan_sections(
        features.sections,
        features=features,
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        difficulty=0.5,
        sync_strength=0.75,
    )
    features.confidence_report.overall = 0.1
    low = _plan_sections(
        features.sections,
        features=features,
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        difficulty=0.5,
        sync_strength=0.75,
    )

    assert max(plan.trigger_intensity for plan in low) <= max(plan.trigger_intensity for plan in high)


def test_section_boundaries_are_not_too_dense(tmp_path: Path) -> None:
    audio_path = tmp_path / "clicks.wav"
    _write_click_wav(audio_path, duration=4.0)

    features = analyze_audio(audio_path, target_duration=4.0, backend="fallback")

    assert len(features.section_boundaries) <= 8


def test_section_planner_outputs_valid_sections(tmp_path: Path) -> None:
    audio_path = tmp_path / "clicks.wav"
    _write_click_wav(audio_path, duration=3.0)
    features = analyze_audio(audio_path, target_duration=3.0, backend="fallback")

    plans = _plan_sections(
        features.sections,
        features=features,
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        difficulty=0.5,
        sync_strength=0.75,
    )

    assert plans
    assert all(plan.end_x >= plan.start_x for plan in plans)
    assert all(0.0 <= plan.density_target <= 1.0 for plan in plans)


def test_scoring_returns_normalized_scores(tmp_path: Path) -> None:
    audio_path = tmp_path / "clicks.wav"
    _write_click_wav(audio_path, duration=2.0)
    features = analyze_audio(audio_path, target_duration=2.0, backend="fallback")
    beat_map = build_beat_x_map(features.beats, [], start_speed="normal", song_offset=0.0)
    report = round_trip_error_report(features.beats, beat_map, [], start_speed="normal", song_offset=0.0)
    objects = [f"1,1,2,{int(x)},3,90" for x in beat_map.values()]

    score = compute_audio_conditioned_score(
        objects,
        audio_features=features,
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=50,
    ).to_dict()

    for key, value in score.items():
        if key == "total":
            assert value >= 0.0
        else:
            assert 0.0 <= value <= 1.0
    assert report["max_error"] < 1e-6


def test_group_id_allocator_bounds() -> None:
    allocator = GroupAllocator(max_group_id=2)
    assert allocator.allocate() == 1
    assert allocator.allocate() == 2
    try:
        allocator.allocate()
    except ValueError as exc:
        assert "group id budget exhausted" in str(exc)
    else:
        raise AssertionError("allocator should raise after max_group_id")


def test_audio_conditioned_generation_creates_valid_gmd(tmp_path: Path) -> None:
    audio_path = tmp_path / "clicks.wav"
    _write_click_wav(audio_path, duration=2.5)

    result = generate_from_config(
        {
            "audio_file": str(audio_path),
            "output_dir": str(tmp_path / "outputs"),
            "output_name": "audio_level",
            "target_duration": 2.5,
            "object_budget": 80,
            "difficulty": "normal",
            "start_speed": "normal",
            "song_offset": 0.0,
            "allow_speed_portals": True,
            "speed_portal_policy": "musical",
            "allow_triggers": True,
            "enforce_quality_gate": False,
            "max_events_per_beat": 2,
            "beat_snap_tolerance": 0.1,
            "onset_event_threshold": 0.2,
        }
    )

    output_path = Path(result["output_path"])
    assert output_path.exists()
    assert result["generation_mode"] == "audio_conditioned"
    assert result["audio_backend"] in {"fallback_wav", "librosa"}
    assert result["valid"] is True
    assert result["num_objects"] > 0
    assert result["score"]["time_to_x_consistency"] > 0.95
    assert parse_gmd_file(output_path).tags["k4"][1]
