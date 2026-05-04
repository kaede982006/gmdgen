# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from gmdgen.generate.audio_conditioned import _maybe_apply_ai_provider
from gmdgen.audio.analysis import AudioAnalysisResult
from gmdgen.gd.plans import SectionPlan

def test_no_improvement_early_stopping():
    config = {
        "use_ai_planner": True,
        "ai_candidate_count": 10,
        "stop_if_no_improvement_rounds": 2,
        "min_quality_improvement_delta": 0.05,
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
    
    # Mock AI provider that returns same response every time
    mock_provider = MagicMock()
    from gmdgen.ai.schemas import AILevelPlanResponse
    mock_provider.generate_level_plan.return_value = AILevelPlanResponse(provider="mock", model="static")
    
    with patch("gmdgen.generate.audio_conditioned.create_ai_provider_from_config", return_value=mock_provider):
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
        
        # Should have stopped after 3 rounds (round 1 best, round 2 same, round 3 same -> stop)
        assert metadata.get("stopped_reason") is not None
        assert "no_improvement" in metadata["stopped_reason"]
        assert len(metadata.get("candidate_reports", [])) < 10
