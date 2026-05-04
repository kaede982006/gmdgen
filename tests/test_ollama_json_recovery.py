# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock
from gmdgen.ai.ollama_provider import extract_json_object, OllamaInvalidJSON, OllamaProvider
from gmdgen.ai.schemas import AILevelPlanResponse
from gmdgen.generate.audio_conditioned import _maybe_apply_ai_provider
from gmdgen.audio.analysis import AudioAnalysisResult

def test_extract_json_valid():
    raw = '{"sections": [], "gameplay_events": []}'
    parsed = extract_json_object(raw)
    assert isinstance(parsed, dict)
    assert "sections" in parsed

def test_extract_json_with_fences():
    raw = 'Here is the JSON:\n```json\n{"sections": [{"start_time": 0}]}\n```\nHope it helps!'
    parsed = extract_json_object(raw)
    assert isinstance(parsed, dict)
    assert parsed["sections"][0]["start_time"] == 0

def test_extract_json_with_text():
    raw = 'Some text before {"a": 1} some text after'
    parsed = extract_json_object(raw)
    assert parsed == {"a": 1}

def test_extract_json_array():
    raw = 'List: [{"id": 1}, {"id": 2}] end'
    parsed = extract_json_object(raw)
    assert isinstance(parsed, list)
    assert len(parsed) == 2

def test_extract_json_trailing_comma():
    raw = '{"a": 1, "b": 2,}'
    parsed = extract_json_object(raw)
    assert parsed == {"a": 1, "b": 2}

def test_extract_json_smart_quotes():
    raw = '\u201ca\u201d: \u201cvalue\u201d'
    # This raw string is not a valid JSON yet, it's "a": "value" with smart quotes.
    # extract_json_object should normalize it to "a": "value" and then try parsing.
    # But "a": "value" without braces is still not valid JSON.
    # We should wrap it in braces to test normalization inside an object.
    raw = '{\u201ca\u201d: \u201cvalue\u201d}'
    parsed = extract_json_object(raw)
    assert parsed == {"a": "value"}

def test_extract_json_invalid_raises():
    with pytest.raises(OllamaInvalidJSON):
        extract_json_object("not json at all")

def test_ollama_provider_repair_retry():
    # Mock requests to fail once with invalid JSON, then succeed
    mock_client = MagicMock()
    # First call returns invalid JSON
    # Second call (repair) returns valid JSON
    mock_client.side_effect = [
        {"response": "Invalid JSON {"},
        {"response": '{"sections": [], "metadata": {"repaired": true}}'}
    ]
    
    provider = OllamaProvider(model="test-model", client=mock_client)
    # We use a dummy debug_dir to trigger _save_debug_artifact if needed, but not required
    
    result = provider._post("test prompt")
    assert result["metadata"]["repaired"] is True
    assert mock_client.call_count == 2

def test_deterministic_fallback_on_json_error():
    config = {
        "use_ai_planner": True,
        "require_ai_planning": False,
        "ollama_client": lambda p: {"response": "Invalid JSON {"}
    }
    
    features = MagicMock(spec=AudioAnalysisResult)
    features.sections = []
    features.beat_times = []
    features.onset_times = []
    features.onset_envelope = []
    features.duration = 10.0
    features.bpm = 120.0
    features.confidence = 0.8
    features.confidence_report = None
    features.beat_features = []
    
    conversion, metadata = _maybe_apply_ai_provider(
        config=config,
        features=features,
        section_plans=[],
        time_x_report={},
        style_profile={},
        object_budget=100,
        max_group_id=999,
        safe_mode=True,
        start_speed="normal",
        song_offset=0.0
    )
    
    assert conversion.valid is True
    assert conversion.response.provider == "local"
    assert conversion.response.fallback_used is True
    assert metadata["deterministic_fallback_used"] is True
    assert metadata["ai_planning_error"] == "ollama_invalid_json"
    assert any("Ollama AI plan was invalid JSON" in w for w in conversion.warnings)

def test_raise_on_json_error_when_required():
    config = {
        "use_ai_planner": True,
        "require_ai_planning": True,
        "ollama_client": lambda p: {"response": "Invalid JSON {"}
    }
    
    features = MagicMock(spec=AudioAnalysisResult)
    features.sections = []
    features.beat_times = []
    features.onset_times = []
    features.onset_envelope = []
    features.duration = 10.0
    features.bpm = 120.0
    features.confidence = 0.8
    features.confidence_report = None
    features.beat_features = []
    
    from gmdgen.errors import ProviderError
    with pytest.raises(ProviderError) as excinfo:
        _maybe_apply_ai_provider(
            config=config,
            features=features,
            section_plans=[],
            time_x_report={},
            style_profile={},
            object_budget=100,
            max_group_id=999,
            safe_mode=True,
            start_speed="normal",
            song_offset=0.0
        )
    assert excinfo.value.code == "ollama_invalid_json"
