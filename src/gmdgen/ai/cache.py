# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ContextCacheRecord:
    source_hash: str
    updated_at: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_context_cache(cache_path: str | Path, record: ContextCacheRecord) -> Path:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_context_cache(cache_path: str | Path) -> ContextCacheRecord | None:
    path = Path(cache_path)
    if not path.exists() or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ContextCacheRecord(
        source_hash=str(payload.get("source_hash", "")),
        updated_at=str(payload.get("updated_at", "")),
        payload=dict(payload.get("payload", {})),
    )


@dataclass(slots=True)
class AIRequestBudget:
    max_calls: int = 2
    calls_used: int = 0
    provider_calls: dict[str, int] = field(default_factory=dict)
    stopped_by_budget: bool = False
    warnings: list[str] = field(default_factory=list)

    def can_call(self) -> bool:
        return self.calls_used < max(0, int(self.max_calls))

    def record_call(self, provider_name: str) -> None:
        if not self.can_call():
            self.stopped_by_budget = True
            warning = f"AI call budget exhausted: {self.calls_used}/{self.max_calls}"
            if warning not in self.warnings:
                self.warnings.append(warning)
            raise RuntimeError(warning)
        self.calls_used += 1
        provider = str(provider_name or "unknown")
        self.provider_calls[provider] = self.provider_calls.get(provider, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AIResponseCacheEntry:
    request_hash: str
    provider_name: str
    model_name: str
    schema_hash: str
    prompt_hash: str
    response_path: str
    created_at: str
    ttl_seconds: int = 86400
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AIResponseCache:
    def __init__(self, cache_dir: str | Path = "dataset/cache/ai_responses", *, ttl_seconds: int = 86400) -> None:
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = int(ttl_seconds)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def build_request_hash(
        self,
        *,
        provider_name: str,
        model_name: str,
        schema: dict[str, Any],
        prompt_payload: Any,
    ) -> tuple[str, str, str]:
        schema_text = _stable_json(schema)
        prompt_text = _stable_json(prompt_payload)
        schema_hash = _sha256(schema_text)
        prompt_hash = _sha256(prompt_text)
        request_hash = _sha256(
            _stable_json(
                {
                    "provider_name": provider_name,
                    "model_name": model_name,
                    "schema_hash": schema_hash,
                    "prompt_hash": prompt_hash,
                }
            )
        )
        return request_hash, schema_hash, prompt_hash

    def load(self, request_hash: str) -> dict[str, Any] | None:
        path = self._entry_path(request_hash)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def save(
        self,
        *,
        request_hash: str,
        provider_name: str,
        model_name: str,
        schema_hash: str,
        prompt_hash: str,
        response_payload: dict[str, Any],
    ) -> AIResponseCacheEntry:
        response_path = self.cache_dir / f"{request_hash}.json"
        payload = {
            "entry": {
                "request_hash": request_hash,
                "provider_name": provider_name,
                "model_name": model_name,
                "schema_hash": schema_hash,
                "prompt_hash": prompt_hash,
                "response_path": response_path.name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "ttl_seconds": self.ttl_seconds,
                "cache_hit": False,
            },
            "response_payload": _sanitize_cache_payload(response_payload),
        }
        response_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return AIResponseCacheEntry(**payload["entry"])

    def clear(self) -> int:
        count = 0
        if not self.cache_dir.exists():
            return 0
        for path in self.cache_dir.glob("*.json"):
            try:
                path.unlink()
                count += 1
            except OSError:
                continue
        return count

    def _entry_path(self, request_hash: str) -> Path:
        return self.cache_dir / f"{request_hash}.json"


def _stable_json(value: Any) -> str:
    return json.dumps(_sanitize_cache_payload(value), ensure_ascii=False, sort_keys=True, default=str)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _sanitize_cache_payload(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if "api_key" in lowered or "base_url" in lowered or "host" in lowered:
                continue
            sanitized[key] = _sanitize_cache_payload(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_cache_payload(item) for item in value]
    if isinstance(value, str):
        return _redact_key_like_text(value)
    return value


def _redact_key_like_text(text: str) -> str:
    value = str(text)
    for marker in ("sk-", "AIza"):
        if marker in value:
            value = value.replace(marker, f"{marker}[REDACTED]-")
    return value
