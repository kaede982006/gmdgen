# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from gmdgen.ai.cache import AIRequestBudget
except Exception:
    AIRequestBudget = None  # type: ignore

from gmdgen.ai.ollama_provider import OllamaOnlyConfigurationError, OllamaProvider

ConfigurationError = OllamaOnlyConfigurationError

@dataclass
class AIProviderAuditResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ai_provider: str = "ollama"
    provider_chain: list[str] = field(default_factory=lambda: ["ollama"])
    local_fallback_used: bool = False
    ollama_available: bool = True

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

def _effective_provider(config: dict[str, Any]) -> str:
    raw = config.get("ai_provider", config.get("primary_provider", "ollama"))
    if raw is None or str(raw).strip() == "":
        return "ollama"
    return str(raw).strip().lower()

def _has_forbidden_key(config: dict[str, Any], prefix: str) -> str | None:
    for key, value in config.items():
        lowered = str(key).lower()
        if lowered.startswith(prefix) and value not in (None, "", [], {}):
            return str(key)
    return None

def _fallback_provider_names(config: dict[str, Any], primary: str = "ollama") -> list[str]:
    raw = config.get("fallback_providers", [])
    if raw in (None, "", [], ()):
        return []
    if isinstance(raw, str):
        names = [item.strip().lower() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, (list, tuple, set)):
        names = [str(item).strip().lower() for item in raw if str(item).strip()]
    else:
        names = [str(raw).strip().lower()]
    return [name for name in names if name and name != primary]

def create_ai_provider_from_config(config: dict[str, Any] | None = None) -> OllamaProvider:
    cfg = dict(config or {})

    provider = _effective_provider(cfg)
    if provider != "ollama":
        raise OllamaOnlyConfigurationError(
            f"Unsupported ai_provider '{provider}'. Only 'ollama' is supported."
        )

    bad_key = _has_forbidden_key(cfg, "gemini_") or _has_forbidden_key(cfg, "openai_")
    if bad_key:
        raise OllamaOnlyConfigurationError(
            f"{bad_key} is not allowed in Ollama-only mode."
        )

    fallbacks = _fallback_provider_names(cfg, primary="ollama")
    if fallbacks:
        raise OllamaOnlyConfigurationError(
            f"fallback_providers is not supported in Ollama-only mode: {fallbacks}"
        )

    budget = None
    if AIRequestBudget is not None and bool(cfg.get("enforce_ai_call_budget", True)):
        try:
            budget = AIRequestBudget(max_calls=int(cfg.get("max_ai_calls_per_generation", 2)))
        except Exception:
            budget = None

    return OllamaProvider(
        model=str(cfg.get("ollama_model", "qwen2.5-coder:7b")),
        base_url=str(cfg.get("ollama_base_url", "http://127.0.0.1:11434")),
        client=cfg.get("ollama_client"),
        timeout_seconds=float(cfg.get("ollama_timeout_seconds", cfg.get("ai_timeout_seconds", 60.0))),
        max_retries=int(cfg.get("ollama_max_retries", cfg.get("ai_retry_count", 1))),
        max_output_tokens=(
            int(cfg["max_output_tokens"])
            if cfg.get("max_output_tokens") is not None
            else None
        ),
        num_ctx=(
            int(cfg["ollama_num_ctx"])
            if cfg.get("ollama_num_ctx") is not None
            else int(cfg.get("ai_num_ctx", 4096))
        ),
        cache_enabled=bool(cfg.get("cache_ai_responses", True)),
        cache_dir=str(cfg.get("ai_response_cache_dir", "dataset/cache/ai_responses")),
        request_budget=budget,
        debug_dir=(
            str(cfg.get("ollama_debug_dir") or (str(cfg.get("output_dir", "outputs")).rstrip("/") + "/debug"))
            if bool(cfg.get("ollama_save_debug_artifacts", False) or cfg.get("save_debug_bundle", False))
            else None
        ),
    )

def create_provider_from_config(config: dict[str, Any] | None = None) -> OllamaProvider:
    return create_ai_provider_from_config(config)

def create_ai_provider(config: dict[str, Any] | None = None) -> OllamaProvider:
    return create_ai_provider_from_config(config)

def audit_ai_provider_config(config: dict[str, Any] | None = None) -> AIProviderAuditResult:
    errors: list[str] = []
    try:
        create_ai_provider_from_config(config or {})
    except Exception as exc:
        errors.append(str(exc))
    return AIProviderAuditResult(passed=not errors, errors=errors)

def run_ollama_only_audit(config: dict[str, Any] | None = None) -> AIProviderAuditResult:
    return audit_ai_provider_config(config)
