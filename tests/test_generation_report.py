from __future__ import annotations

import math
import struct
import wave
from pathlib import Path
import json

from gmdgen.generate.generator import generate_from_config


def _write_click_wav(path: Path) -> None:
    sample_rate = 8000
    duration = 2.0
    samples = [0.0] * int(sample_rate * duration)
    for beat in [0.0, 0.5, 1.0, 1.5]:
        start = int(beat * sample_rate)
        for idx in range(start, min(len(samples), start + int(0.02 * sample_rate))):
            phase = (idx - start) / max(1, int(0.02 * sample_rate))
            samples[idx] = math.sin(phase * math.pi) * 0.9
    pcm = b"".join(
        struct.pack("<h", max(-32768, min(32767, int(sample * 32767))))
        for sample in samples
    )
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm)


def test_generation_report_contains_mode(tmp_path: Path) -> None:
    result = generate_from_config(
        {
            "output_dir": str(tmp_path / "outputs"),
            "output_name": "style_only",
            "num_objects": 8,
        }
    )
    assert result["generation_mode"] == "style_only"


def test_audio_conditioned_report_contains_audio_stats(tmp_path: Path) -> None:
    audio_path = tmp_path / "clicks.wav"
    _write_click_wav(audio_path)
    result = generate_from_config(
        {
            "audio_file": str(audio_path),
            "audio_backend": "fallback",
            "output_dir": str(tmp_path / "outputs"),
            "output_name": "audio_report",
            "target_duration": 2.0,
            "object_budget": 40,
        }
    )
    for key in ["audio_backend", "bpm", "num_beats", "num_onsets", "num_sections"]:
        assert key in result


def test_validation_report_contains_score_breakdown(tmp_path: Path) -> None:
    audio_path = tmp_path / "clicks.wav"
    _write_click_wav(audio_path)
    result = generate_from_config(
        {
            "audio_file": str(audio_path),
            "audio_backend": "fallback",
            "output_dir": str(tmp_path / "outputs"),
            "output_name": "score_report",
            "target_duration": 2.0,
            "object_budget": 40,
        }
    )
    assert result["validation_report"]["score_breakdown"]


def test_validation_report_contains_time_x_errors(tmp_path: Path) -> None:
    audio_path = tmp_path / "clicks.wav"
    _write_click_wav(audio_path)
    result = generate_from_config(
        {
            "audio_file": str(audio_path),
            "audio_backend": "fallback",
            "output_dir": str(tmp_path / "outputs"),
            "output_name": "time_x_report",
            "target_duration": 2.0,
            "object_budget": 40,
        }
    )
    report = result["validation_report"]
    assert "time_x_avg_error" in report
    assert "time_x_max_error" in report


def test_generation_metadata_contains_audio_conditioned_fields(tmp_path: Path) -> None:
    audio_path = tmp_path / "clicks.wav"
    _write_click_wav(audio_path)
    result = generate_from_config(
        {
            "audio_file": str(audio_path),
            "audio_backend": "fallback",
            "output_dir": str(tmp_path / "outputs"),
            "output_name": "metadata_audio",
            "target_duration": 2.0,
            "object_budget": 40,
        }
    )
    for key in [
        "generation_mode",
        "audio_backend",
        "bpm",
        "num_beats",
        "num_onsets",
        "num_sections",
        "num_speed_portals",
        "num_objects",
        "num_triggers",
        "time_x_average_error",
        "time_x_max_error",
        "final_score",
    ]:
        assert key in result


def test_generation_metadata_contains_editor_safety_fields(tmp_path: Path) -> None:
    audio_path = tmp_path / "clicks.wav"
    _write_click_wav(audio_path)
    result = generate_from_config(
        {
            "audio_file": str(audio_path),
            "audio_backend": "fallback",
            "output_dir": str(tmp_path / "outputs"),
            "output_name": "metadata_safety",
            "target_duration": 2.0,
            "object_budget": 40,
        }
    )
    assert "editor_safety" in result
    assert "round_trip_valid" in result
    assert "playability_warning_count" in result
    assert "trigger_schema_warning_count" in result


def test_generation_metadata_serializable(tmp_path: Path) -> None:
    audio_path = tmp_path / "clicks.wav"
    _write_click_wav(audio_path)
    result = generate_from_config(
        {
            "audio_file": str(audio_path),
            "audio_backend": "fallback",
            "output_dir": str(tmp_path / "outputs"),
            "output_name": "metadata_json",
            "target_duration": 2.0,
            "object_budget": 40,
        }
    )
    json.dumps(result, sort_keys=True)
