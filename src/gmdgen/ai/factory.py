# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import os
import logging
from typing import Any

from gmdgen.ai.gemini_provider import GeminiProvider
from gmdgen.ai.provider import LevelGenerationAIProvider

logger = logging.getLogger(__name__)

def create_ai_provider_from_config(config: dict[str, Any] | None = None) -> LevelGenerationAIProvider:
    cfg = dict(config or {})
    provider_name = str(cfg.get("ai_provider", os.environ.get("GMDGEN_PROVIDER", "gemini"))).lower()
    
    if provider_name in ["openai", "ollama", "qwen", "qwen3"]:
        raise ValueError(f"Legacy provider '{provider_name}' is no longer supported. Please use 'gemini'.")
        
    return GeminiProvider(
        model=cfg.get("gemini_model", cfg.get("model", "gemini-2.5-flash")),
        api_key=cfg.get("gemini_api_key", os.environ.get("GEMINI_API_KEY")),
        timeout_seconds=float(cfg.get("ai_timeout_seconds", 60.0)),
        max_retries=int(cfg.get("ai_retry_count", 2))
    )

def create_provider_from_config(config: dict[str, Any] | None = None) -> LevelGenerationAIProvider:
    return create_ai_provider_from_config(config)
