# SPDX-License-Identifier: GPL-3.0-or-later
import json
import os
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

def test_audio_conditioned_without_ai_planning_is_deterministic_and_no_ai(tmp_path: Path):
    audio_path = tmp_path / "clicks.wav"
    _write_click_wav(audio_path)
    
    config = {
        "audio_file": str(audio_path),
        "audio_backend": "fallback",
        "output_dir": str(tmp_path / "outputs"),
        "output_name": "no_ai_test",
        "target_duration": 2.0,
        "object_budget": 40,
        "use_ai_planner": False
    }
    
    result = generate_from_config(config)
    
    assert result["generation_mode"] == "audio_conditioned"
    assert result.get("ai_used") is not True
    assert "local" in result.get("ai_provider", "local")
    assert result["num_sections"] >= 1
    assert "validation_report" in result
    
    json_str = json.dumps(result)
    assert json_str

def test_audio_conditioned_with_ai_planning_fails_if_no_gemini_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    audio_path = tmp_path / "clicks.wav"
    _write_click_wav(audio_path)
    
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    
    config = {
        "audio_file": str(audio_path),
        "audio_backend": "fallback",
        "output_dir": str(tmp_path / "outputs"),
        "output_name": "ai_test",
        "target_duration": 2.0,
        "object_budget": 40,
        "use_ai_planner": True,
        "require_ai_planning": True,
    }

    with pytest.raises(Exception, match="GEMINI_API_KEY"):
        generate_from_config(config)
