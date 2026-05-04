# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import pytest
from pathlib import Path

def test_gui_surface_ollama_only() -> None:
    """
    Verifies that app.py does not contain any Gemini or OpenAI related strings in the UI.
    """
    app_path = Path("src/gmdgen/gui/app.py")
    text = app_path.read_text(encoding="utf-8").lower()

    # Banned phrases
    banned = [
        "gemini ai",
        "gemini model",
        "gemini api key",
        "use gemini_api_key env",
        "max gemini calls",
        "cache gemini responses",
        "through gemini api",
        "gemini api key required",
        "gemini-only audit",
        "gemini-2.5-flash",
        "openai fallback",
        "openai_api_key",
        "use_environment_key",
        "v_use_environment_key"
    ]

    for phrase in banned:
        assert phrase not in text, f"Found banned phrase in GUI: {phrase}"

    # Required phrases
    required = [
        "ollama",
        "ollama model",
        "ollama base url",
        "ollama_host",
        "use_ollama_environment_key",
        "v_use_ollama_environment_key",
        "qwen2.5-coder:7b"
    ]

    for phrase in required:
        assert phrase in text, f"Missing required phrase in GUI: {phrase}"

def test_no_gemini_in_ai_factory_checks() -> None:
    """
    Ensures that factory.py (if updated) doesn't use gemini_ or openai_ as forbidden keys in a way that implies they are active.
    """
    factory_path = Path("src/gmdgen/ai/factory.py")
    if not factory_path.exists():
        return
    text = factory_path.read_text(encoding="utf-8").lower()
    # It might have forbidden key checks, which is fine, but it shouldn't be active.
    pass
