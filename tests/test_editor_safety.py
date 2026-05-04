from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from gmdgen.gd.plans import ObjectPlan, TriggerMode, TriggerPlan
from gmdgen.generate.editor_safety import (
    editor_safety_score,
    run_encoder_round_trip_safety_check,
    validate_editor_safety,
    validate_save_string_safety,
)
from gmdgen.generate.generator import generate_from_config


def _write_click_wav(path: Path) -> None:
    sample_rate = 8000
    duration = 2.0
    samples = [0.0] * int(sample_rate * duration)
    for beat in [0.0, 0.5, 1.0, 1.5]:
        start = int(beat * sample_rate)
        for idx in range(start, min(len(samples), start + int(0.02 * sample_rate))):
            samples[idx] = math.sin(((idx - start) / 160) * math.pi) * 0.8
    pcm = b"".join(
        struct.pack("<h", max(-32768, min(32767, int(sample * 32767))))
        for sample in samples
    )
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm)


def test_editor_safety_rejects_nan_coordinates() -> None:
    report = validate_save_string_safety("kA11,0;1,1,2,nan,3,180;", TriggerMode.SAFE)
    assert report.valid is False
    assert report.nan_coordinate_count > 0


def test_editor_safety_rejects_unknown_trigger_in_safe_mode() -> None:
    report = validate_save_string_safety("kA11,0;1,1347,2,30,3,180,51,1;", TriggerMode.SAFE)
    assert report.valid is False
    assert report.unsupported_trigger_count > 0


def test_editor_safety_detects_group_bounds() -> None:
    report = validate_save_string_safety("kA11,0;1,1,2,30,3,180,155,999;", TriggerMode.SAFE, max_group_id=10)
    assert report.valid is False
    assert report.invalid_group_count > 0


def test_editor_safety_round_trip_object_count() -> None:
    report = validate_save_string_safety("kA11,0;1,1,2,30,3,180;1,1,2,60,3,180;", TriggerMode.SAFE)
    assert report.round_trip_ok is True
    assert report.object_count_before == report.object_count_after


def test_editor_safety_penalizes_invalid_candidate() -> None:
    good = validate_save_string_safety("kA11,0;1,1,2,30,3,180;", TriggerMode.SAFE)
    bad = validate_save_string_safety("kA11,0;1,901,2,nan,3,180;", TriggerMode.SAFE)
    assert editor_safety_score(bad) < editor_safety_score(good)


def test_validate_editor_safety_from_plans() -> None:
    report = run_encoder_round_trip_safety_check(
        [ObjectPlan("1", 30, 180, "structure", group_ids=[1])],
        [TriggerPlan("pulse", "1006", x=60, y=240, target_group=1, duration=0.2)],
        TriggerMode.SAFE,
        max_group_id=10,
    )
    assert report.valid is True


def test_audio_conditioned_final_candidate_has_editor_safety_report(tmp_path: Path) -> None:
    audio_path = tmp_path / "clicks.wav"
    _write_click_wav(audio_path)
    result = generate_from_config(
        {
            "audio_file": str(audio_path),
            "audio_backend": "fallback",
            "output_dir": str(tmp_path / "outputs"),
            "output_name": "safe_audio",
            "target_duration": 2.0,
            "object_budget": 45,
        }
    )
    assert result["generation_mode"] == "audio_conditioned"
    assert result["editor_safety"]
    assert result["editor_safety"]["valid"] is True
