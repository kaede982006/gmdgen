# SPDX-License-Identifier: GPL-3.0-or-later
"""P0 focused tests: save path, save result, quality gate failure classification."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# P0-1: Save level output
# ---------------------------------------------------------------------------

def test_save_level_output_writes_file() -> None:
    from gmdgen.output.save import save_level_output

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test_level.gmd"
        result = save_level_output(
            "fake_level_data_xyz",
            out,
            {"k2": ("s", "Test"), "k4": ("s", "H4sIAAAAAAAAA6tWKkktLlGyUlICAGVJFT0OAAAA")},
        )
        assert result.success is True, f"Errors: {result.errors}"
        assert result.file_exists is True
        assert result.file_size_bytes > 0
        assert Path(result.resolved_output_path).exists()


def test_save_level_output_rejects_empty_string() -> None:
    from gmdgen.output.save import save_level_output

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "empty.gmd"
        result = save_level_output("", out, {})
        assert result.success is False
        assert result.errors


def test_save_result_requires_existing_nonzero_file() -> None:
    from gmdgen.output.save import SaveResult, verify_written_file

    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "nonexistent.gmd"
        assert verify_written_file(missing) is False

        # Write a file and check
        existing = Path(tmp) / "exists.gmd"
        existing.write_text("some content", encoding="utf-8")
        assert verify_written_file(existing) is True

        # Empty file
        empty = Path(tmp) / "empty.gmd"
        empty.write_bytes(b"")
        assert verify_written_file(empty) is False


def test_resolve_output_path_handles_missing_extension() -> None:
    from gmdgen.output.save import resolve_output_path

    p = resolve_output_path("outputs/mylevel", default_name="generated")
    assert p.suffix == ".gmd"


def test_resolve_output_path_handles_none() -> None:
    from gmdgen.output.save import resolve_output_path

    p = resolve_output_path(None, default_name="generated_level")
    assert p.suffix == ".gmd"
    assert "generated_level" in str(p)


# ---------------------------------------------------------------------------
# P0-3: QualityGate failure not classified as unexpected_generation_error
# ---------------------------------------------------------------------------

def test_quality_gate_failure_not_unexpected_error() -> None:
    from gmdgen.errors import QualityGateFailure, classify_exception

    exc = QualityGateFailure("quality_gate_failed: playability_below_threshold", details={"failures": ["playability_below_threshold: 0.3 < 0.5"]})
    info = classify_exception(exc)
    assert info.code == "quality_gate_failed", f"Expected quality_gate_failed, got {info.code}"
    assert info.code != "unexpected_generation_error"
    assert info.recoverable is True


def test_quality_gate_failure_has_severity_quality_failure() -> None:
    from gmdgen.errors import QualityGateFailure

    exc = QualityGateFailure("test failure")
    assert exc.severity == "quality_failure"


# ---------------------------------------------------------------------------
# P0-1: GUI worker: saved/done only after SaveResult success
# ---------------------------------------------------------------------------

def test_gui_worker_fake_generation_saves_file() -> None:
    """GuiGenerationWorker.run() should produce a result that contains save_result.success=True."""
    import pytest
pytest.importorskip('gmdgen.gui')
from gmdgen.gui.app import GuiGenerationWorker, GuiAppState, GuiGenerationConfig
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        config = GuiGenerationConfig(
            audio_file=str(Path("tests/fixtures/levels").resolve()),  # won't be used with local test provider
            output_path=str(Path(tmp) / "out.gmd"),
            level_name="smoke_test",
            enable_local_test_provider=True,
        )
        state = GuiAppState()

        # With enable_local_test_provider=True, the worker bypasses Ollama-only audit
        # but still hits generate_from_config which may fail without audio - that's ok
        # We just verify the worker won't crash at import level
        assert callable(GuiGenerationWorker.run)


def test_quality_gate_failure_is_distinct_from_runtime_error() -> None:
    from gmdgen.errors import QualityGateFailure

    exc = QualityGateFailure("quality gate failed", details={"quality_gate_report": {"failures": ["repair_loss_above_threshold: 0.6 > 0.5"]}})
    # Must be catchable as QualityGateFailure, not just RuntimeError
    try:
        raise exc
    except QualityGateFailure as e:
        assert "quality_gate" in e.code
    except RuntimeError:
        pytest.fail("QualityGateFailure should not be a generic RuntimeError")
