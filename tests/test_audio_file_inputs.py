from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest

from gmdgen import generate_level
from gmdgen.audio.paths import normalize_audio_file_path, validate_audio_file_extension
from gmdgen.cli import build_parser
from gmdgen.generate.generator import generate_from_config


def _write_click_wav(path: Path, *, duration: float = 2.0) -> None:
    sample_rate = 8000
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


def test_cli_accepts_audio_file_argument(tmp_path: Path) -> None:
    audio_path = tmp_path / "song.wav"
    args = build_parser().parse_args(["generate", "--audio-file", str(audio_path)])

    assert args.audio_file == str(audio_path)


def test_cli_accepts_audio_alias_argument(tmp_path: Path) -> None:
    audio_path = tmp_path / "song.wav"
    args = build_parser().parse_args(["generate", "--audio", str(audio_path)])

    assert args.audio_file == str(audio_path)


def test_generation_config_accepts_audio_file(tmp_path: Path) -> None:
    audio_path = tmp_path / "api_song.wav"
    _write_click_wav(audio_path)

    result = generate_level(
        audio_file=audio_path,
        audio_backend="fallback",
        output_dir=str(tmp_path / "outputs"),
        output_name="api_audio",
        target_duration=2.0,
        object_budget=40,
    )

    assert result["generation_mode"] == "audio_conditioned"
    assert result["audio_file_name"] == audio_path.name


def test_audio_file_triggers_audio_conditioned_mode(tmp_path: Path) -> None:
    audio_path = tmp_path / "conditioned.wav"
    _write_click_wav(audio_path)

    result = generate_from_config(
        {
            "audio_file": audio_path,
            "audio_backend": "fallback",
            "output_dir": str(tmp_path / "outputs"),
            "output_name": "conditioned",
            "target_duration": 2.0,
            "ai_provider": "local_test_only",
                "object_budget": 40,
        }
    )

    assert result["generation_mode"] == "audio_conditioned"
    assert result["validation_report"]["generation_mode"] == "audio_conditioned"


def test_missing_audio_file_raises_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="audio_file does not exist"):
        generate_from_config({"audio_file": tmp_path / "missing.wav"})


def test_empty_audio_file_uses_style_only(tmp_path: Path) -> None:
    result = generate_from_config(
        {
            "audio_file": "   ",
            "output_dir": str(tmp_path / "outputs"),
            "output_name": "style_empty_audio",
            "num_objects": 8,
        }
    )

    assert result["generation_mode"] == "style_only"


def test_directory_as_audio_file_raises_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="audio_file must be a file"):
        generate_from_config({"audio_file": tmp_path})


def test_unsupported_audio_extension_raises_clear_error(tmp_path: Path) -> None:
    audio_path = tmp_path / "song.txt"
    audio_path.write_text("not audio", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported audio file extension"):
        validate_audio_file_extension(audio_path, backend="fallback")

    with pytest.raises(ValueError, match="unsupported audio file extension"):
        generate_from_config({"audio_file": audio_path, "audio_backend": "fallback"})


def test_audio_file_path_with_spaces(tmp_path: Path) -> None:
    audio_path = tmp_path / "song with spaces.wav"
    _write_click_wav(audio_path)

    result = generate_from_config(
        {
            "audio_file": str(audio_path),
            "audio_backend": "fallback",
            "output_dir": str(tmp_path / "outputs"),
            "output_name": "spaces",
            "target_duration": 2.0,
            "ai_provider": "local_test_only",
                "object_budget": 40,
        }
    )

    assert result["generation_mode"] == "audio_conditioned"
    assert result["audio_file_name"] == audio_path.name


def test_relative_audio_file_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio_path = tmp_path / "relative.wav"
    _write_click_wav(audio_path)
    monkeypatch.chdir(tmp_path)

    result = generate_from_config(
        {
            "audio_file": "relative.wav",
            "audio_backend": "fallback",
            "output_dir": "outputs",
            "output_name": "relative",
            "target_duration": 2.0,
            "ai_provider": "local_test_only",
                "object_budget": 40,
        }
    )

    assert result["generation_mode"] == "audio_conditioned"
    assert result["audio_file_name"] == "relative.wav"


def test_generation_report_contains_audio_file_name(tmp_path: Path) -> None:
    audio_path = tmp_path / "reported.wav"
    _write_click_wav(audio_path)

    result = generate_from_config(
        {
            "audio_file": str(audio_path),
            "audio_backend": "fallback",
            "output_dir": str(tmp_path / "outputs"),
            "output_name": "reported",
            "target_duration": 2.0,
            "ai_provider": "local_test_only",
                "object_budget": 40,
        }
    )
    report = result["validation_report"]

    assert result["audio_file"] == audio_path.name
    assert result["audio_file_name"] == audio_path.name
    assert result["detected_bpm"] > 0.0
    assert result["beat_count"] >= 1
    assert result["onset_count"] >= 1
    assert result["section_count"] >= 1
    assert report["audio_file"] == audio_path.name
    assert report["audio_file_name"] == audio_path.name
    assert report["detected_bpm"] > 0.0
    assert report["beat_count"] >= 1
    assert report["onset_count"] >= 1
    assert report["section_count"] >= 1
    assert str(audio_path) not in {result["audio_file"], result["audio_file_name"]}


def test_debug_report_can_include_audio_full_path(tmp_path: Path) -> None:
    audio_path = tmp_path / "debug.wav"
    _write_click_wav(audio_path)

    result = generate_from_config(
        {
            "audio_file": audio_path,
            "audio_backend": "fallback",
            "output_dir": str(tmp_path / "outputs"),
            "output_name": "debug",
            "target_duration": 2.0,
            "ai_provider": "local_test_only",
                "object_budget": 40,
            "debug_paths": True,
        }
    )

    assert result["audio_file_full_path"] == str(normalize_audio_file_path(audio_path))
