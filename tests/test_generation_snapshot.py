# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from gmdgen.generate.generator import generate_from_config


LEVEL_FIXTURES = Path("tests/fixtures/levels")


def _write_audio(path: Path, *, duration: float, bpm: float, profile: str) -> None:
    sample_rate = 8000
    interval = 60.0 / bpm
    samples = []
    for idx in range(int(duration * sample_rate)):
        t = idx / sample_rate
        amp = 0.08 if profile == "steady" else 0.08 + 0.5 * min(1.0, t / duration)
        if profile == "low_high" and t > duration / 2:
            amp = 0.65
        value = math.sin(2 * math.pi * 220 * t) * amp
        if (t % interval) < 0.025:
            value += 0.8
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


def _generate(tmp_path: Path, audio_path: Path, *, output_name: str, reference: str, budget: int = 70) -> dict:
    return generate_from_config(
        {
            "audio_file": str(audio_path),
            "audio_backend": "fallback",
            "output_dir": str(tmp_path / "outputs"),
            "output_name": output_name,
            "style_reference_level": str(LEVEL_FIXTURES / reference),
            "target_duration": 3.0,
            "ai_provider": "local_test_only",
                "object_budget": budget,
            "allow_triggers": True,
            "allow_speed_portals": True,
            "speed_portal_policy": "musical",
            "safe_mode": True,
            "enforce_quality_gate": False,
        }
    )


def test_snapshot_steady_120bpm_audio_conditioned(tmp_path: Path) -> None:
    audio = tmp_path / "steady.wav"
    _write_audio(audio, duration=3.0, bpm=120, profile="steady")
    result = _generate(tmp_path, audio, output_name="steady_snapshot", reference="basic_blocks_reference.gmd")
    assert result["generation_mode"] == "audio_conditioned"
    assert 3 <= result["num_beats"] <= 16
    assert result["num_objects"] <= 70
    assert result["time_x_report"]["max_error"] <= 1e-6


def test_snapshot_buildup_drop_sections(tmp_path: Path) -> None:
    audio = tmp_path / "buildup.wav"
    _write_audio(audio, duration=3.0, bpm=120, profile="buildup")
    result = _generate(tmp_path, audio, output_name="buildup_snapshot", reference="basic_blocks_reference.gmd")
    assert 1 <= result["num_sections"] <= 8
    assert result["analysis_confidence"] >= 0.05


def test_snapshot_reference_level_style_consistency(tmp_path: Path) -> None:
    audio = tmp_path / "style.wav"
    _write_audio(audio, duration=3.0, bpm=120, profile="steady")
    result = _generate(tmp_path, audio, output_name="style_snapshot", reference="trigger_safe_reference.gmd")
    assert result["score"]["style_consistency"] >= 0.0
    assert result["editor_safety"]["valid"] is True


def test_snapshot_speed_portal_reference_time_x(tmp_path: Path) -> None:
    audio = tmp_path / "speed.wav"
    _write_audio(audio, duration=3.0, bpm=120, profile="low_high")
    result = _generate(tmp_path, audio, output_name="speed_snapshot", reference="speed_portal_reference.gmd")
    assert result["time_x_report"]["checked_count"] > 0
    assert result["TimeXConsistencyScore"] >= 0.95


def test_snapshot_no_orphan_triggers(tmp_path: Path) -> None:
    audio = tmp_path / "orphan.wav"
    _write_audio(audio, duration=3.0, bpm=120, profile="steady")
    result = _generate(tmp_path, audio, output_name="orphan_snapshot", reference="trigger_safe_reference.gmd")
    assert result["validation_report"]["orphan_trigger_count"] == 0
    assert result["TriggerValidityScore"] >= 0.99


def test_snapshot_object_budget_respected(tmp_path: Path) -> None:
    audio = tmp_path / "budget.wav"
    _write_audio(audio, duration=3.0, bpm=140, profile="buildup")
    result = _generate(tmp_path, audio, output_name="budget_snapshot", reference="basic_blocks_reference.gmd", budget=35)
    assert result["num_objects"] <= 35
