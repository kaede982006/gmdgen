# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_default_entrypoint_launches_gui_or_gui_main() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    env["GMDGEN_HEADLESS"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "gmdgen"],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert completed.returncode == 0
    assert "context initialized only" in completed.stdout.lower()


def test_cli_generate_is_disabled_for_user_generation(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    env["GMDGEN_HEADLESS"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "gmdgen", "generate", "--audio-file", str(tmp_path / "song.wav")],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert completed.returncode != 0
    assert "GUI-only for user generation" in completed.stderr


def test_cli_does_not_allow_real_generation(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    env["GMDGEN_HEADLESS"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "gmdgen", "generate", "--audio-file", str(tmp_path / "song.wav"), "--ai-provider", "ollama"],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert completed.returncode != 0
    assert "GUI-only for user generation" in completed.stderr


def test_gui_module_imports() -> None:
    import pytest
pytest.importorskip('gmdgen.gui')
import gmdgen.gui.app  # noqa: F401
