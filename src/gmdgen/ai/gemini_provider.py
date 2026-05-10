# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
import logging
import time
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore

from gmdgen.ai.provider import LevelGenerationAIProvider, AIProviderUsage
from gmdgen.ai.schemas import AILevelPlanRequest, AILevelPlanResponse, parse_ai_level_plan_response
from gmdgen.ai.provider_errors import AIProviderError, AIProviderErrorInfo

logger = logging.getLogger(__name__)

class GeminiProvider(LevelGenerationAIProvider):
    provider_name = "gemini"

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        cache_enabled: bool = True,
        cache_dir: str = "dataset/cache/ai_responses",
        **kwargs: Any
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir
        self._client = None
        
        if genai is None:
            # We don't raise here to allow 'gmdgen doctor' to report it gracefully
            pass

    @property
    def client(self):
        if self._client is None:
            if genai is None:
                raise ImportError("google-genai package is required. Install with 'pip install google-genai'.")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY environment variable is required.")
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def classify_error(self, error: Exception) -> AIProviderErrorInfo:
        err_str = str(error).lower()
        code = "unknown"
        exhausted = False
        recoverable = False
        
        if "auth" in err_str or "api_key" in err_str or "401" in err_str or "403" in err_str:
            code = "auth"
        elif "quota" in err_str or "exhausted" in err_str or "429" in err_str:
            code = "provider_exhausted"
            exhausted = True
            recoverable = True
        elif "timeout" in err_str:
            code = "timeout"
            recoverable = True
        elif "network" in err_str or "connection" in err_str:
            code = "network"
            recoverable = True
            
        return AIProviderErrorInfo(
            provider_name=self.provider_name,
            code=code,
            message=str(error),
            recoverable=recoverable,
            provider_exhausted=exhausted
        )

    def generate_level_plan(self, request: AILevelPlanRequest) -> AILevelPlanResponse:
        prompt = "Create a Geometry Dash level plan based on the provided audio features. Return only JSON."
        # Note: In a real implementation, we'd use request.to_prompt() or similar
        
        start_time = time.time()
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                )
            )
            latency = time.time() - start_time
            
            data = json.loads(response.text)
            parsed = parse_ai_level_plan_response(data)
            parsed.provider = self.provider_name
            parsed.model = self.model
            parsed.metadata["latency_seconds"] = latency
            return parsed
            
        except Exception as e:
            info = self.classify_error(e)
            raise AIProviderError(info)

