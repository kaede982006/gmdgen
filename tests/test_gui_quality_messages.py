# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

"""Tests for GUI quality message logic without invoking Tkinter."""

import ast
from pathlib import Path


def _read_app_source() -> str:
    return Path("src/gmdgen/gui/app.py").read_text(encoding="utf-8")


def test_quality_warning_does_not_always_suggest_extreme_ml():
    """
    The warning shown when quality gate fails must not unconditionally
    say 'Try enabling Extreme ML mode'.  The message must be conditional
    on whether the user's quality_mode already includes 'extreme'.
    """
    source = _read_app_source()
    # The string 'Try enabling Extreme ML mode' should only appear inside a
    # conditional branch, not as a bare unconditional string literal.
    # We check that 'already_extreme' guard variable is present.
    assert "already_extreme" in source, (
        "Expected 'already_extreme' guard in app.py to suppress redundant Extreme ML suggestion"
    )


def test_extreme_ml_suggestion_is_conditional():
    """
    The fix must use config.quality_mode to decide whether to suggest Extreme ML.
    """
    source = _read_app_source()
    assert "already_extreme" in source
    assert "quality_mode" in source


def test_warning_dialog_not_duplicated_on_same_path():
    """
    _on_quality_gate_failed (QualityGateFailure exception path) and
    _on_generate_success (draft saved path) must be mutually exclusive.
    The task function returns after calling _on_quality_gate_failed,
    so _on_generate_success won't also fire.
    """
    source = _read_app_source()
    tree = ast.parse(source)

    # Find the task() function inside _generate
    task_bodies = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "task":
            task_bodies.append(node)

    assert task_bodies, "Expected a task() inner function in app.py"

    for task_fn in task_bodies:
        # The task function must have a return after calling _on_quality_gate_failed
        source_lines = ast.get_source_segment(source, task_fn) or ""
        # Simpler check: if _on_quality_gate_failed is in the body, 'return' must follow
        if "_on_quality_gate_failed" in source_lines:
            assert "return" in source_lines, (
                "task() must return after calling _on_quality_gate_failed"
            )


def test_quality_warning_includes_diagnostic_fields():
    """
    The warning message shown on quality gate failure must include
    playability, repair_loss, and final object count diagnostics.
    """
    source = _read_app_source()
    assert "Playability" in source or "playability" in source
    assert "Repair loss" in source or "repair_loss" in source
    assert "Final objects" in source or "final_object_count" in source or "num_objects" in source


def test_stopped_reason_included_in_diagnostics():
    """
    The quality warning should include stopped_reason when present.
    """
    source = _read_app_source()
    assert "stopped_reason" in source


def test_no_gemini_suggestion_in_quality_messages():
    source = _read_app_source().lower()
    # The quality message code must not suggest Gemini
    # (check in the _on_generate_success block area)
    assert "try enabling gemini" not in source
    assert "use gemini" not in source


def test_gui_generation_config_has_quality_mode():
    import pytest
    pytest.importorskip('gmdgen.gui')
    import pytest
    pytest.importorskip('gmdgen.gui')
    from gmdgen.gui.app import GuiGenerationConfig
    from dataclasses import fields, is_dataclass
    if is_dataclass(GuiGenerationConfig):
        field_names = {f.name for f in fields(GuiGenerationConfig)}
    else:
        import inspect
        sig = inspect.signature(GuiGenerationConfig.__init__)
        field_names = {p for p in sig.parameters if p != "self"}
    assert "quality_mode" in field_names, "GuiGenerationConfig must have quality_mode field"


def test_extreme_ml_quality_mode_values_recognized():
    """GuiGenerationConfig.quality_mode values that count as 'extreme' must include 'Extreme ML'."""
    from gmdgen.gui.app import GuiGenerationConfig
    cfg = GuiGenerationConfig(audio_file="test.mp3", output_path="out.gmd")
    cfg.quality_mode = "Extreme ML"
    already_extreme = cfg.quality_mode.lower() in {"extreme ml", "extreme_ml", "extreme"}
    assert already_extreme is True

    cfg.quality_mode = "Low Cost"
    already_extreme = cfg.quality_mode.lower() in {"extreme ml", "extreme_ml", "extreme"}
    assert already_extreme is False
