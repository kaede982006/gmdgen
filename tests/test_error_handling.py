# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from gmdgen.errors import (
    classify_exception,
    format_error_for_gui,
    format_error_for_log,
    redact_text,
    should_abort_generation,
    should_retry,
)


def test_generation_error_classification() -> None:
    info = classify_exception(RuntimeError("unsupported_object_role[0]: jump_pad"))

    assert info.code == "ai_plan_validation_error"
    assert should_abort_generation(info) is True
    assert "AI output validation failed" in format_error_for_gui(info)


def test_ollama_missing_key_classified() -> None:
    info = classify_exception(RuntimeError("Ollama base URL or OLLAMA_HOST is required"))

    assert info.code == "ollama_base_url_missing"
    assert "Ollama base URL or OLLAMA_HOST is required" in info.user_message


def test_ollama_timeout_retried() -> None:
    info = classify_exception(TimeoutError("request timed out"))

    assert info.code == "ollama_timeout"
    assert should_retry(info) is True


def test_invalid_ai_json_retried_or_reported() -> None:
    info = classify_exception(RuntimeError("ollama_generation_failed: invalid JSON"))

    assert info.code in {"unexpected_generation_error", "ai_plan_validation_error"}
    assert format_error_for_log(info)


def test_api_key_redacted_from_all_error_outputs(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-123456")
    info = classify_exception(RuntimeError("bad key sk-test-secret-123456"))

    assert "sk-test-secret-123456" not in format_error_for_gui(info)
    assert "sk-test-secret-123456" not in format_error_for_log(info)
    assert "sk-test-secret-123456" not in redact_text("bad sk-test-secret-123456")
