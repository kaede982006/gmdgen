# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

import pytest

from gmdgen.gui.app import GuiApplication, GuiGenerationConfig, GuiGenerationWorker, redact_text, safe_gui_callback
from gmdgen.learning.store import load_learning_examples
from gmdgen.learning.feature_extractor import load_learned_data_store


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


def test_gui_worker_requires_external_ai_for_real_generation(tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    _write_click_wav(audio)
    app = GuiApplication()
    config = GuiGenerationConfig(
        audio_file=str(audio),
        output_path=str(tmp_path / "out.gmd"),
        use_ollama_environment_key=False,
    )
    with pytest.raises(ValueError, match="Ollama base URL or OLLAMA_HOST is required"):
        app.generate(config)


def test_gui_worker_uses_auto_training_context(tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    _write_click_wav(audio)
    app = GuiApplication()
    state = app.startup()
    assert state.training_result is not None
    config = GuiGenerationConfig(
        audio_file=str(audio),
        output_path=str(tmp_path / "out.gmd"),
        use_ollama_environment_key=False,
        enable_local_test_provider=True,
        context_dir="docs",
    )
    generated = config.to_generation_config()
    assert generated["context_dir"] == "docs"


def test_gui_worker_blocks_generation_when_audit_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = tmp_path / "song.wav"
    _write_click_wav(audio)
    app = GuiApplication()
    config = GuiGenerationConfig(
        audio_file=str(audio),
        output_path=str(tmp_path / "out.gmd"),
        use_ollama_environment_key=False,
        ollama_base_url="sk-[REDACTED]",
    )
    
    # Mock generate_from_config to raise an error, simulating audit failure
    def raise_generation_error(*args, **kwargs):
        raise RuntimeError("Ollama-only audit failed")
    
    monkeypatch.setattr("gmdgen.gui.app.generate_from_config", raise_generation_error)
    with pytest.raises(RuntimeError, match="Ollama-only audit failed"):
        app.generate(config)


def test_gui_worker_redacts_api_key_in_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = tmp_path / "song.wav"
    _write_click_wav(audio)
    app = GuiApplication()
    config = GuiGenerationConfig(
        audio_file=str(audio),
        output_path=str(tmp_path / "out.gmd"),
        use_ollama_environment_key=False,
        ollama_base_url="sk-[REDACTED]",
        enable_local_test_provider=True,
    )
    worker = GuiGenerationWorker(app.state, config)
    monkeypatch.setattr("gmdgen.gui.app.generate_from_config", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad sk-[REDACTED]")))
    with pytest.raises(RuntimeError):
        worker.run()
    assert "sk-secret" not in worker.error


def test_gui_worker_returns_validation_report_with_many_values(tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    _write_click_wav(audio)
    app = GuiApplication()
    config = GuiGenerationConfig(
        audio_file=str(audio),
        output_path=str(tmp_path / "out.gmd"),
        enable_local_test_provider=True,
        difficulty="harder",
        sync_strength=0.9,
        speed_portal_policy="musical",
        max_events_per_beat=3,
        enforce_quality_gate=False,
    )
    result = app.generate(config)
    assert "validation_report" in result
    assert result["validation_report"]["ai_required"] is True
    assert result["ai_provider"] == "local_test_only"
    assert result["generation_mode"] == "audio_conditioned"


def test_gui_worker_auto_saves_learning_example(tmp_path: Path) -> None:
    audio = tmp_path / "learn.wav"
    _write_click_wav(audio)
    app = GuiApplication()
    config = GuiGenerationConfig(
        audio_file=str(audio),
        output_path=str(tmp_path / "learn.gmd"),
        enable_local_test_provider=True,
        save_learning_data=True,
        learning_store_dir=str(tmp_path / "learning"),
        enforce_quality_gate=False,
    )

    result = app.generate(config)
    records = load_learning_examples(store_dir=tmp_path / "learning")

    assert result["learning_example_id"]
    assert app.state.last_learning_example_id == result["learning_example_id"]
    assert len(records) == 1
    assert "ollama_base_url" not in json.dumps(records[0])


def test_gui_worker_does_not_save_learning_when_disabled(tmp_path: Path) -> None:
    audio = tmp_path / "no_learn.wav"
    _write_click_wav(audio)
    app = GuiApplication()
    config = GuiGenerationConfig(
        audio_file=str(audio),
        output_path=str(tmp_path / "no_learn.gmd"),
        enable_local_test_provider=True,
        save_learning_data=False,
        learning_store_dir=str(tmp_path / "learning"),
        enforce_quality_gate=False,
    )

    result = app.generate(config)

    assert "learning_example_id" not in result
    assert load_learning_examples(store_dir=tmp_path / "learning") == []


def test_learn_data_worker_extracts_features(tmp_path: Path) -> None:
    level = tmp_path / "ref.gmd"
    level.write_text("kA11,0;1,1,2,0,3,90;1,500,2,60,3,240;1,36,2,120,3,180;", encoding="utf-8")
    app = GuiApplication()

    result = app.learn_data(str(level), store_dir=str(tmp_path / "learned"))
    store = load_learned_data_store(store_dir=tmp_path / "learned")

    assert result["learned_level_count"] == 1
    assert result["extracted_motif_count"] >= 1
    assert len(store.learned_levels) == 1


def test_learn_data_worker_handles_invalid_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.gmd"
    bad.write_text("not a real level", encoding="utf-8")
    app = GuiApplication()

    result = app.learn_data(str(bad), store_dir=str(tmp_path / "learned"))

    assert result["failure_pattern_count"] >= 1


def test_safe_gui_callback_catches_exception() -> None:
    class Dummy:
        def __init__(self) -> None:
            self.handled = ""

        def _handle_gui_exception(self, exc: Exception, *, stage: str = "callback") -> None:
            self.handled = f"{stage}:{exc}"

        @safe_gui_callback
        def boom(self) -> None:
            raise RuntimeError("bad sk-secret")

    dummy = Dummy()

    dummy.boom()

    assert dummy.handled.startswith("boom:")


def test_gui_error_boundary_redacts_api_key() -> None:
    assert "sk-secret" not in redact_text("failure sk-secret happened")


def test_gui_report_contains_quality_fields(tmp_path: Path) -> None:
    audio = tmp_path / "quality.wav"
    _write_click_wav(audio)
    app = GuiApplication()
    config = GuiGenerationConfig(
        audio_file=str(audio),
        output_path=str(tmp_path / "quality.gmd"),
        enable_local_test_provider=True,
        ai_candidate_count=2,
        min_acceptable_score=0.4,
        min_drop_impact_score=0.2,
        enforce_quality_gate=False,
    )

    result = app.generate(config)
    report = result["validation_report"]

    assert "plan_snapshots" in report
    assert "plan_diffs" in report
    assert "candidate_reports" in report
    assert "quality_loss_reason_summary" in report
    assert "drop_impact_score" in report
    assert "density_target_error" in report
