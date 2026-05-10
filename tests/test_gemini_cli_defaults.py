# SPDX-License-Identifier: GPL-3.0-or-later
import pytest
from gmdgen.ai.factory import create_ai_provider_from_config
from gmdgen.ai.gemini_provider import GeminiProvider

def test_gemini_is_default():
    provider = create_ai_provider_from_config({})
    assert isinstance(provider, GeminiProvider)

def test_ollama_not_default():
    with pytest.raises(ValueError, match="Legacy provider 'ollama' is no longer supported"):
        create_ai_provider_from_config({"ai_provider": "ollama"})

def test_qwen_not_default():
    with pytest.raises(ValueError, match="Legacy provider 'qwen' is no longer supported"):
        create_ai_provider_from_config({"ai_provider": "qwen"})

def test_localhost_11434_not_default():
    with pytest.raises(ValueError, match="Legacy provider 'ollama' is no longer supported"):
        create_ai_provider_from_config({"ai_provider": "ollama", "ollama_base_url": "http://127.0.0.1:11434"})

def test_openai_fallback_disabled_by_default():
    with pytest.raises(ValueError, match="Legacy provider 'openai' is no longer supported"):
        create_ai_provider_from_config({"ai_provider": "openai"})

def test_openai_fallback_explicit_only():
    with pytest.raises(ValueError, match="Legacy provider 'openai' is no longer supported"):
        create_ai_provider_from_config({"ai_provider": "openai", "allow_fallback": True})
