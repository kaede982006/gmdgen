# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import os
from typing import Any
from gmdgen.ai.gemini_provider import GeminiProvider
from gmdgen.ai.ollama_provider import OllamaProvider
from gmdgen.ai.provider import LevelGenerationAIProvider

def create_ai_provider_from_config(config: dict[str, Any] | None = None) -> LevelGenerationAIProvider:
    cfg = dict(config or {})
    provider_name = str(cfg.get("ai_provider", os.environ.get("GMDGEN_PROVIDER", "gemini"))).lower()
    
    if provider_name == "gemini":
        return GeminiProvider(
            model=cfg.get("gemini_model", cfg.get("model", "gemini-2.5-flash")),
            api_key=cfg.get("gemini_api_key"),
            timeout_seconds=float(cfg.get("ai_timeout_seconds", 60.0)),
            max_retries=int(cfg.get("ai_retry_count", 1))
        )
    elif provider_name == "ollama":
        return OllamaProvider(
            model=str(cfg.get("ollama_model", "qwen2.5-coder:7b")),
            base_url=str(cfg.get("ollama_base_url", "http://127.0.0.1:11434")),
            timeout_seconds=float(cfg.get("ollama_timeout_seconds", 60.0))
        )
    else:
        # Default to Gemini for backward compatibility with user intent
        return GeminiProvider()

def create_provider_from_config(config: dict[str, Any] | None = None) -> LevelGenerationAIProvider:
    return create_ai_provider_from_config(config)

