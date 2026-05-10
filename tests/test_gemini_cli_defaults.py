import os
import pytest
from gmdgen.ai.factory import create_ai_provider_from_config
from gmdgen.ai.gemini_provider import GeminiProvider

def test_gemini_first_default_provider():
    provider = create_ai_provider_from_config({})
    assert isinstance(provider, GeminiProvider), "Gemini should be the default provider"
    assert provider.provider_name == "gemini"

def test_ollama_not_default():
    provider = create_ai_provider_from_config({"ai_provider": "ollama"})
    # It should fall back to Gemini
    assert isinstance(provider, GeminiProvider), "Ollama should not be allowed as default, fallback to Gemini"

def test_qwen_not_default():
    provider = create_ai_provider_from_config({"ai_provider": "qwen"})
    assert isinstance(provider, GeminiProvider)

def test_localhost_11434_not_default():
    # If someone tries to use ollama, ensure base_url is not actively used
    provider = create_ai_provider_from_config({"ai_provider": "ollama", "ollama_base_url": "http://127.0.0.1:11434"})
    assert isinstance(provider, GeminiProvider)

def test_openai_fallback_disabled_by_default():
    provider = create_ai_provider_from_config({"ai_provider": "openai"})
    # Fallback is not enabled, should default to Gemini
    assert isinstance(provider, GeminiProvider)

def test_openai_fallback_explicit_only():
    provider = create_ai_provider_from_config({"ai_provider": "openai", "allow_fallback": True})
    # This should be the OpenAIFallbackProvider stub
    assert provider.__class__.__name__ == "OpenAIFallbackProvider"

def test_gemini_missing_api_key_error():
    # If we initialize GeminiProvider explicitly without an API key
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]
    
    provider = GeminiProvider(api_key=None)
    with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable is required"):
        _ = provider.client
