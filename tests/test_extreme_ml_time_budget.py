from __future__ import annotations

import time
import pytest
from unittest.mock import MagicMock, patch
from gmdgen.generate.audio_conditioned import _maybe_apply_ai_provider
from gmdgen.audio.analysis import AudioAnalysisResult
from gmdgen.gd.plans import SectionPlan

def test_extreme_ml_time_budget():
    config = {
        "use_ai_planner": True,
        "ai_candidate_count": 10,
        "max_extreme_ml_seconds": 0.5,
        "quality_mode": "extreme ml",
        "stop_if_no_improvement_rounds": 100, # Disable early stopping for this test
    }
    
    features = MagicMock(spec=AudioAnalysisResult)
    features.beat_times = [0.0, 0.5, 1.0]
    
    from gmdgen.gd.time_mapping import SpeedState
    section_plans = [
        SectionPlan(
            start_time=0.0, end_time=1.0, start_x=0.0, end_x=100.0,
            section_type="normal", gameplay_mode="cube", speed_state=SpeedState.NORMAL,
            density_target=0.5, decoration_intensity=0.5, trigger_intensity=0.5,
            difficulty_target=0.5
        )
    ]
    
    # Mock AI provider that takes 0.2s per call
    mock_provider = MagicMock()
    def slow_generate(*args, **kwargs):
        time.sleep(0.2)
        from gmdgen.ai.schemas import AILevelPlanResponse
        return AILevelPlanResponse(provider="mock", model="slow")
    
    mock_provider.generate_level_plan.side_effect = slow_generate
    
    with patch("gmdgen.generate.audio_conditioned.create_ai_provider_from_config", return_value=mock_provider):
        start_time = time.time()
        conversion, metadata = _maybe_apply_ai_provider(
            config=config,
            features=features,
            section_plans=section_plans,
            time_x_report={},
            style_profile={},
            object_budget=1000,
            max_group_id=9999,
            safe_mode=True,
            start_speed="normal",
            song_offset=0.0,
        )
        duration = time.time() - start_time
        
        # Should have stopped after ~3 calls (0.6s) because 0.5s limit reached
        assert metadata.get("stopped_reason") is not None
        assert "time_budget_exceeded" in metadata["stopped_reason"]
        assert duration < 1.5 # Reasonable upper bound
        assert len(metadata.get("candidate_reports", [])) < 10
