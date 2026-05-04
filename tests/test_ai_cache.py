# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from pathlib import Path

from gmdgen.ai.cache import AIResponseCache


def test_ai_response_cache_misses_on_schema_change(tmp_path: Path) -> None:
    cache = AIResponseCache(tmp_path)
    req1, schema1, prompt1 = cache.build_request_hash(
        provider_name="ollama",
        model_name="ollama-2.5-flash",
        schema={"type": "object", "properties": {"a": {"type": "string"}}},
        prompt_payload={"prompt": "a"},
    )
    req2, _schema2, _prompt2 = cache.build_request_hash(
        provider_name="ollama",
        model_name="ollama-2.5-flash",
        schema={"type": "object", "properties": {"b": {"type": "string"}}},
        prompt_payload={"prompt": "a"},
    )
    cache.save(
        request_hash=req1,
        provider_name="ollama",
        model_name="ollama-2.5-flash",
        schema_hash=schema1,
        prompt_hash=prompt1,
        response_payload={"sections": []},
    )

    assert cache.load(req1) is not None
    assert cache.load(req2) is None


def test_ai_response_cache_does_not_store_api_key(tmp_path: Path) -> None:
    cache = AIResponseCache(tmp_path)
    req, schema_hash, prompt_hash = cache.build_request_hash(
        provider_name="ollama",
        model_name="ollama-2.5-flash",
        schema={"type": "object"},
        prompt_payload={"api_key": "AIzasecretsecretsecret", "prompt": "test"},
    )
    cache.save(
        request_hash=req,
        provider_name="ollama",
        model_name="ollama-2.5-flash",
        schema_hash=schema_hash,
        prompt_hash=prompt_hash,
        response_payload={"debug": "AIzasecretsecretsecret"},
    )

    text = (tmp_path / f"{req}.json").read_text(encoding="utf-8")
    assert "AIzasecretsecretsecret" not in text
    assert "api_key" not in text
