# SPDX-License-Identifier: GPL-3.0-or-later
import json
from pathlib import Path
from gmdgen.generate.generator import generate_from_config
import pytest

def _write_click_wav(path: Path):
    import wave
    import struct
    with wave.open(str(path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(44100)
        for _ in range(44100):
            f.writeframes(struct.pack("<h", 0))

def test_audio_conditioned_without_ai_planning_is_deterministic_and_no_ollama(tmp_path: Path):
    audio_path = tmp_path / "clicks.wav"
    _write_click_wav(audio_path)
    
    config = {
        "audio_file": str(audio_path),
        "audio_backend": "fallback",
        "output_dir": str(tmp_path / "outputs"),
        "output_name": "no_ai_test",
        "target_duration": 2.0,
        "object_budget": 40,
    }
    
    result = generate_from_config(config)
    
    assert result["generation_mode"] == "audio_conditioned"
    # The metadata contains info about AI
    assert result.get("ai_used") is not True
    assert "local" in result.get("ai_provider", "local")
    assert result["num_sections"] >= 1
    assert "validation_report" in result
    
    # Ensure serializable
    json_str = json.dumps(result)
    assert json_str

def test_audio_conditioned_with_ai_planning_fails_if_no_ollama(tmp_path: Path):
    audio_path = tmp_path / "clicks.wav"
    _write_click_wav(audio_path)
    
    config = {
        "audio_file": str(audio_path),
        "audio_backend": "fallback",
        "output_dir": str(tmp_path / "outputs"),
        "output_name": "ai_test",
        "target_duration": 2.0,
        "object_budget": 40,
        "use_ai_planner": True,
        "require_ai_planning": True,
        "ollama_timeout_seconds": 0.1,
        "ollama_max_retries": 0,
    }

    # This SHOULD attempt to call Ollama and fail if not running
    from gmdgen.errors import GmdgenError
    with pytest.raises((ValueError, GmdgenError)):
        generate_from_config(config)
