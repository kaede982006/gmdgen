# SPDX-License-Identifier: GPL-3.0-or-later
import pytest
import os
from pathlib import Path

def test_docs_ollama_final_requirements():
    readme = Path("README.md")
    if readme.exists():
        text = readme.read_text(encoding="utf-8").lower()
        assert "ollama-only" in text or "ollama" in text
        assert "api key" not in text or "no external api key" in text or "does not require an api key" in text
        
    provider_doc = Path("docs/ollama_provider.md")
    if provider_doc.exists():
        text = provider_doc.read_text(encoding="utf-8")
        assert "ollama_server_unavailable" in text
        assert "ollama_model_missing" in text
        assert "ollama_timeout" in text
        assert "ollama_network_error" in text
        assert "ollama_invalid_json" in text
        assert "ollama_invalid_schema" in text
        assert "ollama_raw_gmd_rejected" in text
        assert "ollama_empty_response" in text
        assert "ollama_unknown_error" in text

    direction_doc = Path("docs/project_direction.md")
    if direction_doc.exists():
        text = direction_doc.read_text(encoding="utf-8")
        assert "Audio Analysis" in text
        assert "Ollama Planning" in text
        assert "Internal Representation" in text
        assert "Materializer" in text
        assert "Repairer" in text
        assert "Validator" in text
        assert "QualityGate" in text
        assert "SaveResult" in text
        assert "GUI" in text
        assert "Local Learning Memory" in text

    troubleshoot_doc = Path("docs/troubleshooting.md")
    if troubleshoot_doc.exists():
        text = troubleshoot_doc.read_text(encoding="utf-8").lower()
        assert "ollama not running" in text or "ollama" in text
        assert "model not found" in text
        assert "7b" in text or "heavy" in text
        assert "runner process terminated" in text
        assert "qualitygate" in text or "quality gate failure" in text
        assert ".gmd" in text or "output not saved" in text
