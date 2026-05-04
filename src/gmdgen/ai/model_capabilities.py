from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ModelCapabilities:
    model_name: str
    supports_temperature: bool = False
    supports_top_p: bool = False
    supports_json_schema: bool = True
    supports_reasoning_effort: bool = False
    supports_max_output_tokens: bool = True
    unsupported_params: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_model_capabilities(model_name: str) -> ModelCapabilities:
    name = (model_name or "").strip().lower()
    if not name:
        return ModelCapabilities(model_name=model_name, unsupported_params=["temperature", "top_p"])

    no_temperature_prefixes = (
        "gpt-5",
        "o1",
        "o3",
        "o4",
    )
    if name.startswith(no_temperature_prefixes):
        return ModelCapabilities(
            model_name=model_name,
            supports_temperature=False,
            supports_top_p=False,
            supports_json_schema=True,
            supports_reasoning_effort=True,
            supports_max_output_tokens=True,
            unsupported_params=["temperature", "top_p"],
        )

    temperature_patterns = (
        r"^gpt-4\.1",
        r"^gpt-4o",
        r"^gpt-4-turbo",
        r"^gpt-3\.5",
    )
    if any(re.search(pattern, name) for pattern in temperature_patterns):
        return ModelCapabilities(
            model_name=model_name,
            supports_temperature=True,
            supports_top_p=True,
            supports_json_schema=True,
            supports_reasoning_effort=False,
            supports_max_output_tokens=True,
        )

    # Safe default for unknown/new models: prefer a minimal Responses request.
    return ModelCapabilities(
        model_name=model_name,
        supports_temperature=False,
        supports_top_p=False,
        supports_json_schema=True,
        supports_reasoning_effort=False,
        supports_max_output_tokens=True,
        unsupported_params=["temperature", "top_p"],
    )


def sanitize_ollama_request_params(model_name: str, params: dict[str, Any]) -> tuple[dict[str, Any], list[str], ModelCapabilities]:
    capabilities = get_model_capabilities(model_name)
    sanitized = dict(params)
    removed: list[str] = []
    if not capabilities.supports_temperature and "temperature" in sanitized:
        sanitized.pop("temperature", None)
        removed.append("temperature")
    if not capabilities.supports_top_p and "top_p" in sanitized:
        sanitized.pop("top_p", None)
        removed.append("top_p")
    for param in capabilities.unsupported_params:
        if param in sanitized:
            sanitized.pop(param, None)
            if param not in removed:
                removed.append(param)
    return sanitized, removed, capabilities


def remove_unsupported_ollama_params(params: dict[str, Any], unsupported_params: list[str]) -> tuple[dict[str, Any], list[str]]:
    sanitized = dict(params)
    removed: list[str] = []
    for param in unsupported_params:
        if param in sanitized:
            sanitized.pop(param, None)
            removed.append(param)
    return sanitized, removed


def extract_unsupported_param_from_error(error: BaseException | str) -> str | None:
    text = str(error)
    patterns = (
        r"Unsupported parameter:\s*'([^']+)'",
        r"Unsupported parameter:\s*\"([^\"]+)\"",
        r"param:\s*([A-Za-z0-9_.-]+)",
        r"parameter ['\"]([^'\"]+)['\"] is not supported",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def retry_without_unsupported_param(error: BaseException | str, params: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    param = extract_unsupported_param_from_error(error)
    if not param or param not in params:
        return dict(params), None
    sanitized = dict(params)
    sanitized.pop(param, None)
    return sanitized, param
