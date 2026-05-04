# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from gmdgen.audio.analysis import analyze_audio
from gmdgen.generate.generator import generate_from_config


def _write_synthetic_wav(path: Path, *, duration: float, bpm: float, profile: str) -> None:
    sample_rate = 8000
    samples: list[float] = []
    beat_interval = 60.0 / bpm
    click_len = int(0.025 * sample_rate)
    total = int(duration * sample_rate)
    for idx in range(total):
        t = idx / sample_rate
        amp = 0.12
        if profile == "low_high":
            amp = 0.06 if t < duration / 2 else 0.55
        elif profile == "buildup_drop":
            amp = 0.08 + min(0.7, t / max(duration, 1e-9)) if t < duration * 0.65 else 0.85
        value = math.sin(2 * math.pi * 220 * t) * amp
        beat_pos = t % beat_interval
        if beat_pos < click_len / sample_rate:
            phase = beat_pos / max(click_len / sample_rate, 1e-9)
            value += math.sin(phase * math.pi) * 0.75
        samples.append(max(-1.0, min(1.0, value)))

    pcm = b"".join(
        struct.pack("<h", max(-32768, min(32767, int(sample * 32767))))
        for sample in samples
    )
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm)


def test_audio_analysis_synthetic_120bpm(tmp_path: Path) -> None:
    audio_path = tmp_path / "steady_120bpm_click.wav"
    _write_synthetic_wav(audio_path, duration=4.0, bpm=120.0, profile="steady")

    features = analyze_audio(audio_path, target_duration=4.0, backend="fallback")

    assert 3.9 <= features.duration <= 4.0
    assert 60.0 <= features.bpm <= 200.0
    assert 4 <= len(features.beat_times) <= 16
    assert len(features.onsets) >= 3
    assert features.rms_envelope
    assert 0.0 <= features.confidence <= 1.0
    assert features.confidence_report is not None


def test_audio_analysis_detects_energy_sections(tmp_path: Path) -> None:
    audio_path = tmp_path / "low_high_energy_sections.wav"
    _write_synthetic_wav(audio_path, duration=5.0, bpm=120.0, profile="low_high")

    features = analyze_audio(audio_path, target_duration=5.0, backend="fallback")

    assert len(features.sections) >= 2
    assert len(features.section_boundaries) <= 8
    assert max(section.energy_peak for section in features.sections) > 0.0
    assert any(section.section_type in {"buildup", "drop", "normal"} for section in features.sections)


def test_audio_conditioned_pipeline_uses_audio_mode(tmp_path: Path) -> None:
    audio_path = tmp_path / "buildup_drop_synthetic.wav"
    _write_synthetic_wav(audio_path, duration=3.0, bpm=120.0, profile="buildup_drop")

    result = generate_from_config(
        {
            "audio_file": str(audio_path),
            "audio_backend": "fallback",
            "output_dir": str(tmp_path / "outputs"),
            "output_name": "audio_fixture_level",
            "target_duration": 3.0,
            "ai_provider": "local_test_only",
                "object_budget": 60,
            "difficulty": "normal",
            "allow_triggers": True,
            "allow_speed_portals": True,
            "speed_portal_policy": "musical",
        }
    )

    assert result["generation_mode"] == "audio_conditioned"
    assert result["audio_backend"] == "fallback_wav"
    assert result["num_sections"] >= 1
    assert result["validation_report"]


def test_audio_conditioned_pipeline_produces_validation_report(tmp_path: Path) -> None:
    audio_path = tmp_path / "steady_120bpm_click.wav"
    _write_synthetic_wav(audio_path, duration=2.5, bpm=120.0, profile="steady")

    result = generate_from_config(
        {
            "audio_file": str(audio_path),
            "audio_backend": "fallback",
            "output_dir": str(tmp_path / "outputs"),
            "output_name": "report_level",
            "target_duration": 2.5,
            "ai_provider": "local_test_only",
                "object_budget": 50,
        }
    )
    report = result["validation_report"]

    assert report["generation_mode"] == "audio_conditioned"
    assert report["score_breakdown"]
    assert "time_x_avg_error" in report
    assert "round_trip_valid" in report


def test_audio_conditioned_respects_object_budget(tmp_path: Path) -> None:
    audio_path = tmp_path / "budget.wav"
    _write_synthetic_wav(audio_path, duration=3.0, bpm=140.0, profile="buildup_drop")

    result = generate_from_config(
        {
            "audio_file": str(audio_path),
            "audio_backend": "fallback",
            "output_dir": str(tmp_path / "outputs"),
            "output_name": "budget_level",
            "target_duration": 3.0,
            "ai_provider": "local_test_only",
                "object_budget": 35,
            "allow_triggers": True,
        }
    )

    assert result["num_objects"] <= 35
