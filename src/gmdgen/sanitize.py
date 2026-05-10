# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import os
from typing import Any


def mask_api_key(api_key: str) -> str:
    if not api_key:
        return ""
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return api_key[:3] + "*" * max(4, len(api_key) - 7) + api_key[-4:]


def redact_secret(value: str) -> str:
    if not value:
        return value
    if value.startswith("sk-"):
        return "sk-[REDACTED]"
    if value.startswith("AIza"):
        return "AIza[REDACTED]"
    return value


def redact_text(text: str) -> str:
    if not text:
        return text
    result = text
    ollama_env_key = os.environ.get("OLLAMA_HOST", "")
    if ollama_env_key:
        result = result.replace(ollama_env_key, "[REDACTED_OLLAMA_HOST]")
    for token in result.split():
        if token.startswith("sk-"):
            result = result.replace(token, "sk-[REDACTED]")
        if token.startswith("AIza"):
            result = result.replace(token, "AIza[REDACTED]")
    return result


def sanitize_report(report: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in report.items():
        if "api_key" in str(key).lower() or "base_url" in str(key).lower() or "host" in str(key).lower():
            continue
        if isinstance(value, str):
            sanitized[key] = redact_text(value)
        elif isinstance(value, dict):
            sanitized[key] = sanitize_report(value)  # type: ignore
        elif isinstance(value, list):
            sanitized[key] = [redact_text(item) if isinstance(item, str) else item for item in value]  # type: ignore
        else:
            sanitized[key] = value
    return sanitized


def sanitize_debug_artifact(data: dict[str, Any]) -> dict[str, Any]:
    return sanitize_report(data)
