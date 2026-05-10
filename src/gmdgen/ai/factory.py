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
    # Legacy/deprecated GUI or older code might request ollama, but default to gemini
    provider_name = str(cfg.get("ai_provider", os.environ.get("GMDGEN_PROVIDER", "gemini"))).lower()
    
    if provider_name == "openai" and cfg.get("allow_fallback"):
        logger.warning("OpenAI fallback is used, but Gemini is recommended.")
        # This is a stub for the fallback
        from gmdgen.ai.provider import LevelGenerationAIProvider
        class OpenAIFallbackProvider(LevelGenerationAIProvider):
            def generate_level_plan(self, request):
                raise NotImplementedError("OpenAI provider stub")
        return OpenAIFallbackProvider()
    
    if provider_name == "ollama":
        logger.warning("Ollama provider is deprecated. Using Gemini instead or failing if explicitly required.")
        # We don't default to ollama.
        # Fallthrough to Gemini

    return GeminiProvider(
        model=cfg.get("gemini_model", cfg.get("model", "gemini-2.5-flash")),
        api_key=cfg.get("gemini_api_key", os.environ.get("GEMINI_API_KEY")),
        timeout_seconds=float(cfg.get("ai_timeout_seconds", 60.0)),
        max_retries=int(cfg.get("ai_retry_count", 2))
    )

def create_provider_from_config(config: dict[str, Any] | None = None) -> LevelGenerationAIProvider:
    return create_ai_provider_from_config(config)
