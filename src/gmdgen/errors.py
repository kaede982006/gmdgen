# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import os
import re
import traceback
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GenerationErrorInfo:
    code: str
    user_message: str
    developer_message: str = ""
    severity: str = "error"
    recoverable: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    remediation: str = ""
    sanitized_traceback: str = ""


class GmdgenError(Exception):
    code = "gmdgen_error"
    user_message = "Generation failed."
    severity = "error"
    recoverable = False
    remediation = "Check Logs / Audit for details."

    def __init__(self, message: str | None = None, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message or self.user_message)
        self.details = details or {}


class ConfigurationError(GmdgenError):
    code = "configuration_error"


class GUIError(GmdgenError):
    code = "gui_error"


class AudioInputError(GmdgenError):
    code = "audio_input_error"


class AudioFileNotFoundError(AudioInputError):
    code = "audio_file_not_found"


class UnsupportedAudioFormatError(AudioInputError):
    code = "unsupported_audio_format"


class AudioDecodeError(AudioInputError):
    code = "audio_decode_error"


class AudioAnalysisError(AudioInputError):
    code = "audio_analysis_error"


class OllamaIntegrationError(GmdgenError):
    code = "ollama_integration_error"


class OllamaBaseURLMissingError(OllamaIntegrationError):
    code = "ollama_base_url_missing"


class OllamaAuthenticationError(OllamaIntegrationError):
    code = "ollama_authentication_error"


class OllamaRateLimitError(OllamaIntegrationError):
    code = "ollama_rate_limit"
    recoverable = True


class OllamaTimeoutError(OllamaIntegrationError):
    code = "ollama_timeout"
    recoverable = True


class OllamaSchemaError(OllamaIntegrationError):
    code = "ollama_schema_error"


class OllamaInvalidResponseError(OllamaIntegrationError):
    code = "ollama_invalid_response"


class OllamaModelError(OllamaIntegrationError):
    code = "ollama_model_error"


class ProviderError(GmdgenError):
    code = "provider_error"

    def __init__(self, message: str | None = None, *, code: str | None = None, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, details=details)
        if code:
            self.code = code


class AIPlanError(GmdgenError):
    code = "ai_plan_error"


class AIPlanParseError(AIPlanError):
    code = "ai_plan_parse_error"


class AIPlanSchemaError(AIPlanError):
    code = "ai_plan_schema_error"


class AIPlanNormalizationError(AIPlanError):
    code = "ai_plan_normalization_error"


class AIPlanValidationError(AIPlanError):
    code = "ai_plan_validation_error"


class TriggerSchemaError(GmdgenError):
    code = "trigger_schema_error"


class PlayabilityValidationError(GmdgenError):
    code = "playability_validation_error"


class TimeMappingError(GmdgenError):
    code = "time_mapping_error"


class EncoderError(GmdgenError):
    code = "encoder_error"


class EditorSafetyError(GmdgenError):
    code = "editor_safety_error"


class GeodeIntegrationError(GmdgenError):
    code = "geode_integration_error"


class GeodeNotAvailableError(GeodeIntegrationError):
    code = "geode_not_available"
    recoverable = True


class GeodeVersionMismatchError(GeodeIntegrationError):
    code = "geode_version_mismatch"


class GeodeBridgeTimeoutError(GeodeIntegrationError):
    code = "geode_bridge_timeout"
    recoverable = True


class GeodeBridgeProtocolError(GeodeIntegrationError):
    code = "geode_bridge_protocol_error"


class GeodeParityMismatchError(GeodeIntegrationError):
    code = "geode_parity_mismatch"


class FileIOGenerationError(GmdgenError):
    code = "file_io_generation_error"


class UnexpectedGenerationError(GmdgenError):
    code = "unexpected_generation_error"

class PromptBuildError(GmdgenError):
    code = "prompt_build_error"

class MotifRetrievalError(GmdgenError):
    code = "motif_retrieval_error"

class QualityGateFailure(GmdgenError):
    code = "quality_gate_failed"
    severity = "quality_failure"
    recoverable = True
    remediation = "Try generating again or saving as a low-quality draft."

    def __init__(self, message: str | None = None, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, details=details)


def redact_text(text: str) -> str:
    result = str(text)
    env_key = os.environ.get("OLLAMA_HOST", "")
    ollama_host = os.environ.get("OLLAMA_HOST", "")
    if env_key:
        result = result.replace(env_key, "[REDACTED_OLLAMA_HOST]")
    if ollama_host:
        result = result.replace(ollama_host, "[REDACTED_OLLAMA_HOST]")
    result = re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "sk-[REDACTED]", result)
    result = re.sub(r"AIza[A-Za-z0-9_\-]{12,}", "AIza[REDACTED]", result)
    return result


def classify_exception(exc: BaseException) -> GenerationErrorInfo:
    text = redact_text(str(exc))
    lower = text.lower()
    code = getattr(exc, "code", "")
    if isinstance(exc, GmdgenError):
        return GenerationErrorInfo(
            code=exc.code,
            user_message=redact_text(str(exc) or exc.user_message),
            developer_message=text,
            severity=exc.severity,
            recoverable=exc.recoverable,
            details=getattr(exc, "details", {}),
            remediation=exc.remediation,
            sanitized_traceback=redact_text("".join(traceback.format_exception(exc))),
        )
    if "ollama base url or ollama_host is required" in lower or "ollama_base_url is required" in lower:
        code = "ollama_base_url_missing"
        user_message = "Ollama base URL or OLLAMA_HOST is required. Set OLLAMA_HOST or enter a URL in the GUI."
        remediation = "Set a valid Ollama base URL and retry."
    elif "ollama base url or ollama_host is required" in lower:
        code = "ollama_base_url_missing"
        user_message = "Ollama base URL or OLLAMA_HOST is required for local generation."
        remediation = "Use Ollama with OLLAMA_HOST or set a valid Ollama base URL."
    elif "invalid_json_schema" in lower or "structured output schema is invalid" in lower:
        code = "ollama_schema_error"
        user_message = _summarize_schema_error(text)
        remediation = "This is a schema bug. Ollama was not the root cause; check developer logs."
    elif "insufficient_quota" in lower or "quota" in lower:
        code = "provider_quota_exhausted"
        user_message = "AI provider quota is exhausted."
        remediation = "Use Ollama as primary provider, lower AI call budget, or retry when quota resets."
    elif "rate limit" in lower or "429" in lower:
        code = "provider_rate_limit"
        user_message = "AI provider rate limit reached."
        remediation = "Retry later, enable Low Cost mode, or lower request volume."
    elif "llama runner process has terminated" in lower:
        code = "ollama_runner_crash"
        user_message = "Ollama runner process crashed (OOM or context overflow)."
        remediation = "Try increasing 'ollama_num_ctx', reducing 'max_output_tokens', or using a smaller model."
    elif "timeout" in lower or "timed out" in lower:
        code = "ollama_timeout"
        user_message = "Ollama request timed out."
        remediation = "Retry or increase the timeout."
    elif "unsupported_object_role" in lower or "unknown_trigger_property" in lower or "invalid_trigger_enum" in lower:
        code = "ai_plan_validation_error"
        user_message = "AI output validation failed after repair."
        remediation = "Review AI normalization warnings and prompt/schema constraints."
    elif "forbidden_planner_field" in lower or "forbidden fields:" in lower:
        code = "ollama_forbidden_field"
        fields = _extract_forbidden_fields(text)
        suffix = f": {', '.join(fields)}" if fields else ""
        user_message = f"Forbidden fields{suffix}"
        remediation = "Ollama returned a non-symbolic planner payload; check planner diagnostics and retry."
    elif "audio_file" in lower or "unsupported audio" in lower or "decode" in lower:
        code = "audio_input_error"
        user_message = "Audio input failed."
        remediation = "Check the audio path and supported format."
    elif isinstance(exc, OSError):
        code = "file_io_generation_error"
        user_message = "File I/O failed during generation."
        remediation = "Check paths, permissions, and locked files."
    else:
        code = code or "unexpected_generation_error"
        user_message = text.splitlines()[0][:900] if text else "Unexpected generation error."
        if code == "quality_gate_failed":
            user_message = getattr(exc, "message", "") or user_message
        remediation = "Check Logs / Audit for full details."
    recoverable = code in {"ollama_rate_limit", "provider_rate_limit", "ollama_timeout", "ollama_timeout", "geode_not_available", "quality_gate_failed", "ollama_runner_crash"}
    return GenerationErrorInfo(
        code=code,
        user_message=user_message,
        developer_message=text,
        recoverable=recoverable,
        remediation=remediation,
        sanitized_traceback=redact_text("".join(traceback.format_exception(exc))),
    )


def _extract_forbidden_fields(text: str) -> list[str]:
    fields: list[str] = []
    explicit = re.search(r"Forbidden fields:\s*([^\n.]+)", text, flags=re.IGNORECASE)
    if explicit:
        for item in explicit.group(1).split(","):
            field = item.strip()
            if field and field not in fields:
                fields.append(field)
    for path in re.findall(r"\$[\w.\[\]]+:forbidden_planner_field", text):
        field = re.split(r"\.|\[", path.split(":", 1)[0])[-1].rstrip("]")
        if field and field not in fields:
            fields.append(field)
    return fields


def sanitize_exception(exc: BaseException) -> GenerationErrorInfo:
    return classify_exception(exc)


def format_error_for_gui(error_info: GenerationErrorInfo) -> str:
    parts = [f"{error_info.user_message}"]
    if error_info.code:
        parts.append(f"Code: {error_info.code}")
    if error_info.remediation:
        parts.append(error_info.remediation)
    return "\n".join(parts)


def format_error_for_log(error_info: GenerationErrorInfo) -> str:
    parts = [
        f"code={error_info.code}",
        f"severity={error_info.severity}",
        f"recoverable={error_info.recoverable}",
        f"user_message={error_info.user_message}",
    ]
    if error_info.developer_message:
        parts.append(f"developer_message={error_info.developer_message}")
    if error_info.sanitized_traceback:
        parts.append(error_info.sanitized_traceback)
    return "\n".join(parts)


def should_retry(error_info: GenerationErrorInfo) -> bool:
    return error_info.code in {"provider_rate_limit", "ollama_timeout", "ollama_timeout", "geode_bridge_timeout"}


def should_abort_generation(error_info: GenerationErrorInfo) -> bool:
    return error_info.code not in {"geode_not_available"}


def _summarize_schema_error(text: str) -> str:
    missing_match = re.search(r"Missing '([^']+)'", text)
    if missing_match:
        return f"Ollama schema error: schema missing required key: {missing_match.group(1)}"
    local_match = re.search(r"(?P<path>[\w.\[\]]+): required is missing keys: \[(?P<keys>[^\]]+)\]", text)
    if local_match:
        keys = local_match.group("keys").replace("'", "").replace('"', "")
        return f"Ollama schema error: schema missing required key(s): {keys} at {local_match.group('path')}"
    return "Ollama schema error: structured output schema is invalid."
