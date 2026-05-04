# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import pytest

import json
import math
import os
import struct
import subprocess
import sys
import wave
from pathlib import Path

from gmdgen.generate.generator import generate_from_config


class _FakeResponses:
    def create(self, **kwargs: object) -> object:
        return {
            "output_text": json.dumps(
                {
                    "sections": [],
                    "object_plans": [
                        {
                            "object_id": 500,
                            "x": 150,
                            "y": 240,
                            "role": "visual_accent_target",
                            "group_ids": [1],
                        }
                    ],
                    "trigger_plans": [
                        {"trigger_type": "alpha", "x": 150, "y": 300, "target_group": 1, "duration": 0.1}
                    ],
                    "reasoning_summary": "fake ollama smoke",
                }
            )
        }


class _FakeClient:
    responses = _FakeResponses()


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


def test_smoke_audio_conditioned_local_provider(tmp_path: Path) -> None:
    audio_path = tmp_path / "local_smoke.wav"
    _write_click_wav(audio_path)
    result = generate_from_config(
        {
            "audio_file": audio_path,
            "audio_backend": "fallback",
            "ai_provider": "local_test_only",
            "allow_local_test_provider": True,
            "output_dir": str(tmp_path / "outputs"),
            "output_name": "local_smoke",
            "target_duration": 2.0,
            "object_budget": 40,
        }
    )

    assert result["generation_mode"] == "audio_conditioned"
    assert result["ai_provider"] == "local_test_only"
    assert result["validation_report"]
    assert result["final_score"] >= 0.0
    assert result["num_objects"] > 0
    assert result["editor_safety"]["fatal_errors"] == []


@pytest.mark.skip(reason="Legacy fake Ollama smoke retired; Ollama-only smoke is covered by tests/test_audio_conditioned_ollama.py")
def test_smoke_audio_conditioned_fake_ollama_provider(tmp_path: Path) -> None:
    audio_path = tmp_path / "fake_ollama.wav"
    _write_click_wav(audio_path)
    result = generate_from_config(
        {
            "audio_file": audio_path,
            "audio_backend": "fallback",
            "ai_provider": "ollama",
            "ollama_client": _FakeClient(),
            "ollama_fallback_to_local": False,
            "output_dir": str(tmp_path / "outputs"),
            "output_name": "fake_ollama_smoke",
            "target_duration": 2.0,
            "object_budget": 40,
        }
    )

    assert result["generation_mode"] == "audio_conditioned"
    assert result["ai_used"] is True
    assert result["ai_output_trigger_count"] == 1
    assert result["editor_safety"]["fatal_errors"] == []


def test_smoke_cli_audio_file_local_provider(tmp_path: Path) -> None:
    audio_path = tmp_path / "cli.wav"
    _write_click_wav(audio_path)
    config_path = tmp_path / "generate.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"output_dir: {json.dumps(str(tmp_path / 'cli_outputs'))}",
                "output_name: cli_smoke",
                "audio_backend: fallback",
                "target_duration: 2.0",
                "object_budget: 35",
                "safe_mode: true",
            ]
        ),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gmdgen",
            "generate",
            "--config",
            str(config_path),
            "--audio-file",
            str(audio_path),
            "--test-local-provider",
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=True,
    )
    result = json.loads(completed.stdout)

    assert result["generation_mode"] == "audio_conditioned"
    assert result["ai_provider"] == "local_test_only"
    assert result["num_objects"] > 0
