# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    import requests
except Exception:
    requests = None

try:
    from gmdgen.ai.provider import LevelGenerationAIProvider
except Exception:
    class LevelGenerationAIProvider:
        pass


class OllamaProviderError(RuntimeError):
    code = "ollama_unknown_error"


class OllamaOnlyConfigurationError(ValueError):
    code = "ollama_only_configuration_error"


class OllamaServerUnavailable(OllamaProviderError):
    code = "ollama_server_unavailable"


class OllamaModelMissing(OllamaProviderError):
    code = "ollama_model_missing"


class OllamaTimeout(OllamaProviderError):
    code = "ollama_timeout"


class OllamaNetworkError(OllamaProviderError):
    code = "ollama_network_error"


class OllamaInvalidJSON(OllamaProviderError):
    code = "ollama_invalid_json"


class OllamaInvalidSchema(OllamaProviderError):
    code = "ollama_invalid_schema"


class OllamaRawGMDRejected(OllamaProviderError):
    code = "ollama_raw_gmd_rejected"


class OllamaEmptyResponse(OllamaProviderError):
    code = "ollama_empty_response"


class OllamaUnknownError(OllamaProviderError):
    code = "ollama_unknown_error"


def _looks_like_raw_gmd(text: str) -> bool:
    compact = str(text or "").strip()
    if not compact:
        return False
    if compact.startswith("{") or compact.startswith("[") or "```json" in compact.lower():
        return False

    comma_count = compact.count(",")
    semicolon_count = compact.count(";")

    if semicolon_count >= 3 and comma_count >= 8:
        return True
    if comma_count >= 12 and re.search(r"(^|[;,])\s*1\s*,\s*\d+", compact):
        return True
    if "kS" in compact and "kA" in compact and comma_count >= 8:
        return True

    return False


def extract_json_object(text: str) -> dict[str, Any]:
    if text is None:
        raise OllamaEmptyResponse("Ollama returned None")

    raw = str(text).strip()
    if not raw:
        raise OllamaEmptyResponse("Ollama returned an empty response")

    if _looks_like_raw_gmd(raw):
        raise OllamaRawGMDRejected("raw .gmd/save string rejected")

    # Robust extraction
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE | re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE | re.IGNORECASE)
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Normalize smart quotes and common typos using unicode escapes
    raw = raw.replace("\u201c", "\"").replace("\u201d", "\"").replace("\u2018", "'").replace("\u2019", "'")
    
    # Remove trailing commas before closing braces/brackets
    raw = re.sub(r",\s*([\]}])", r"\1", raw)

    # Extract first balanced object or array
    start_obj = raw.find("{")
    start_arr = raw.find("[")
    
    start = -1
    end_char = ""
    if start_obj != -1 and (start_arr == -1 or start_obj < start_arr):
        start = start_obj
        end_char = "}"
    elif start_arr != -1:
        start = start_arr
        end_char = "]"

    if start != -1:
        depth = 0
        end = -1
        open_char = "{" if end_char == "}" else "["
        for index, ch in enumerate(raw[start:], start=start):
            if ch == open_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end != -1:
            raw_candidate = raw[start:end]
            try:
                return json.loads(raw_candidate)
            except json.JSONDecodeError:
                pass

    # Last resort: try literal_eval if it looks like Python dict/list (conservative)
    if raw.startswith(("{", "[")):
        try:
            import ast
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (dict, list)):
                return parsed
        except (ValueError, SyntaxError, MemoryError, OverflowError):
            pass

    raise OllamaInvalidJSON("no valid JSON object found in response")


@dataclass
class OllamaProvider(LevelGenerationAIProvider):
    model: str = "gmdgen-coder"
    base_url: str = "http://localhost:11434"
    client: Any = None
    timeout_seconds: float = 60.0
    max_retries: int = 1
    max_output_tokens: int | None = None
    num_ctx: int | None = None
    cache_enabled: bool = True
    cache_dir: str = "dataset/cache/ai_responses"
    request_budget: Any = None
    debug_dir: str | None = None

    def __init__(
        self,
        model: str = "gmdgen-coder",
        base_url: str = "http://localhost:11434",
        client: Any = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 1,
        max_output_tokens: int | None = None,
        num_ctx: int | None = None,
        cache_enabled: bool = True,
        cache_dir: str = "dataset/cache/ai_responses",
        request_budget: Any = None,
        debug_dir: str | None = None,
        **_: Any,
    ) -> None:
        self.model = model or "gmdgen-coder"
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")
        self.client = client
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.max_output_tokens = max_output_tokens
        self.num_ctx = num_ctx
        self.cache_enabled = bool(cache_enabled)
        self.cache_dir = cache_dir
        self.request_budget = request_budget
        self.debug_dir = debug_dir

    @staticmethod
    def parse_response(text: str) -> dict[str, Any]:
        data = extract_json_object(text)
        if "response" in data and isinstance(data["response"], str):
            try:
                return extract_json_object(data["response"])
            except OllamaProviderError:
                return data
        return data

    def _consume_budget(self) -> None:
        if self.request_budget is None:
            return
        for name in ("consume", "check", "use"):
            func = getattr(self.request_budget, name, None)
            if callable(func):
                func()
                return

    def _save_debug_artifact(self, name: str, content: str) -> None:
        if not self.debug_dir:
            return
        try:
            debug_path = Path(self.debug_dir)
            debug_path.mkdir(parents=True, exist_ok=True)
            (debug_path / name).write_text(content, encoding="utf-8")
        except Exception:
            pass

    def _post(self, prompt: str, retry_repair: bool = True) -> dict[str, Any]:
        self._consume_budget()

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }

        options: dict[str, Any] = {}
        if self.max_output_tokens is not None:
            options["num_predict"] = int(self.max_output_tokens)
        if self.num_ctx is not None:
            options["num_ctx"] = int(self.num_ctx)
        
        if options:
            payload["options"] = options

        attempts = 0
        max_attempts = max(1, self.max_retries + 1)
        
        last_exc: Exception | None = None

        while attempts < max_attempts:
            attempts += 1
            try:
                response_text = ""
                if self.client is not None:
                    if callable(self.client):
                        raw = self.client(payload)
                        if isinstance(raw, Mapping):
                            if "response" in raw:
                                response_text = str(raw["response"])
                            else:
                                return dict(raw)
                        else:
                            response_text = str(raw)
                    elif hasattr(self.client, "generate"):
                        raw = self.client.generate(**payload)
                        if isinstance(raw, Mapping):
                            if "response" in raw:
                                response_text = str(raw["response"])
                            else:
                                return dict(raw)
                        else:
                            response_text = str(raw)
                    elif hasattr(self.client, "post"):
                        response = self.client.post(
                            f"{self.base_url}/api/generate",
                            json=payload,
                            timeout=self.timeout_seconds,
                        )
                        response_text = getattr(response, "text", "")
                    else:
                        raise OllamaNetworkError("unsupported client object")
                else:
                    if requests is None:
                        raise OllamaNetworkError("requests is not installed")
                    response = requests.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=self.timeout_seconds,
                    )
                    response_text = getattr(response, "text", "")

                status = getattr(response, "status_code", 200) if 'response' in locals() else 200
                if status == 404:
                    raise OllamaModelMissing(response_text or "model missing")
                if status >= 500:
                    # Check for runner termination error
                    if "llama runner process has terminated" in response_text:
                        if attempts < max_attempts:
                            time.sleep(1.0 * attempts) # Exponential backoff
                            continue
                    raise OllamaServerUnavailable(response_text or "server unavailable")
                
                try:
                    return self.parse_response(response_text)
                except OllamaInvalidJSON as exc:
                    self._save_debug_artifact("raw_ollama_response.txt", response_text)
                    self._save_debug_artifact("json_parse_error.txt", str(exc))
                    
                    if retry_repair:
                        repair_prompt = (
                            "Convert the following malformed response into valid JSON only. "
                            "Do not add explanation. Do not use markdown. "
                            "All property names must use double quotes.\n\n"
                            f"Response to repair:\n{response_text}"
                        )
                        try:
                            repaired_data = self._post(repair_prompt, retry_repair=False)
                            self._save_debug_artifact("repaired_ollama_response.txt", json.dumps(repaired_data, indent=2))
                            return repaired_data
                        except Exception as repair_exc:
                            self._save_debug_artifact("repair_failed.txt", str(repair_exc))
                            raise exc
                    raise

            except OllamaProviderError:
                raise
            except Exception as exc:
                last_exc = exc
                lower = f"{exc.__class__.__name__}: {exc}".lower()
                
                # Retry on connection issues
                if attempts < max_attempts and ("connection" in lower or "connect" in lower or "timeout" in lower):
                    time.sleep(0.5 * attempts)
                    continue

                if "timeout" in lower:
                    raise OllamaTimeout(str(exc)) from exc
                if "connection" in lower or "connect" in lower:
                    raise OllamaServerUnavailable(str(exc)) from exc
                raise OllamaNetworkError(str(exc)) from exc
        
        # If we reached here, it means we exhausted retries
        if last_exc:
            raise last_exc
        raise OllamaUnknownError("failed after max retries")

    def generate(self, prompt: str = "", **kwargs: Any) -> dict[str, Any]:
        return self._post(prompt or str(kwargs.get("prompt", "")))

    def generate_plan(self, prompt: str = "", **kwargs: Any) -> dict[str, Any]:
        return self.generate(prompt, **kwargs)

    def generate_level_plan(self, request: Any) -> Any:
        from gmdgen.ai.schemas import AILevelPlanResponse
        from gmdgen.ai.planner import parse_ollama_section_plan
        try:
            req_dict = request.to_dict()
        except AttributeError:
            req_dict = request
            
        prompt = json.dumps(req_dict, ensure_ascii=False)
        raw_dict = self._post(prompt)
        planner_result = parse_ollama_section_plan(raw_dict)
        if not planner_result.valid:
            raise OllamaInvalidSchema("; ".join(planner_result.errors))
        
        return AILevelPlanResponse(
            sections=[],
            gameplay_events=[],
            object_plans=[],
            trigger_plans=[],
            speed_plan=[],
            reasoning_summary="Ollama returned a strict symbolic section plan.",
            safety_notes=raw_dict.get("safety_notes", []),
            expected_sync_notes=raw_dict.get("expected_sync_notes", []),
            metadata={
                "planner_schema": "strict_section_plan_v1",
                "planner_report": planner_result.to_report_fields(),
                "level_plan": raw_dict.get("level_plan", {}),
                "sections": raw_dict.get("sections", []),
            },
            provider="ollama",
            model=self.model,
        )

    def complete(self, prompt: str = "", **kwargs: Any) -> dict[str, Any]:
        return self.generate(prompt, **kwargs)

    def create(self, prompt: str = "", **kwargs: Any) -> dict[str, Any]:
        return self.generate(prompt, **kwargs)
