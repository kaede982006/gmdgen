# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import pytest

from gmdgen.ai.factory import create_ai_provider_from_config
from gmdgen.ai.ollama_provider import (
    OllamaInvalidJSON,
    OllamaProvider,
    OllamaRawGMDRejected,
    extract_json_object,
)

def test_extract_plain_json():
    assert extract_json_object('{"sections": [], "objects": []}')["sections"] == []

def test_extract_fenced_json():
    assert extract_json_object('```json\n{"sections": [], "triggers": []}\n```')["triggers"] == []

def test_extract_balanced_json_inside_text():
    assert extract_json_object('Here is the plan: {"sections": [], "objects": []} done.')["objects"] == []

def test_reject_invalid_json():
    with pytest.raises(OllamaInvalidJSON):
        extract_json_object("{bad json")

def test_reject_raw_gmd_like_text():
    raw = "1,100,2,200,3,300;1,100,2,200,3,300;1,100,2,200,3,300;1,100,2,200,3,300;"
    with pytest.raises(OllamaRawGMDRejected):
        extract_json_object(raw)

def test_fake_client_response_parsed():
    provider = OllamaProvider(client=lambda payload: {"response": "{\"sections\": [], \"objects\": []}"})
    assert provider.generate("make plan")["sections"] == []
