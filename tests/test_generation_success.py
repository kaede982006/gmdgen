# SPDX-License-Identifier: GPL-3.0-or-later
import json
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

from gmdgen.generate.audio_conditioned import generate_audio_synced_level_from_config
from gmdgen.generate.ir import AILevelPlan, AISectionPlan
from gmdgen.ai.schemas import AILevelPlanResponse

class TestGenerationSuccessPath(unittest.TestCase):
    @patch("gmdgen.generate.audio_conditioned.create_ai_provider_from_config")
    @patch("gmdgen.generate.audio_conditioned.analyze_audio")
    @patch("gmdgen.io.gmd_writer.write_gmd_file")
    @patch("gmdgen.generate.audio_conditioned.validate_gmd_file")
    def test_ai_sections_are_applied(self, mock_validate, mock_write, mock_analyze, mock_create_provider):
        mock_validate.return_value = (True, [])
        # 1. Setup mock audio features
        mock_features = MagicMock()
        mock_features.duration = 30.0
        mock_features.bpm = 120.0
        mock_features.beat_times = [float(i) * 0.5 for i in range(60)]
        mock_features.onset_times = [float(i) * 0.5 for i in range(60)]
        mock_features.sections = [] # Simple case
        mock_features.confidence_report = None
        mock_features.backend = "mock"
        mock_analyze.return_value = mock_features

        # 2. Setup mock AI provider
        mock_provider = MagicMock()
        mock_create_provider.return_value = mock_provider
        
        # Define a specific AI plan with a unique game mode to verify application
        ai_sections = [
            AISectionPlan(
                section_id="ai_s1",
                time_start=0.0,
                time_end=30.0,
                game_mode="spider", # Unique mode
                speed="2x",
                density=0.8,
                primary_pattern="ai_test_pattern"
            )
        ]
        ai_level_plan = AILevelPlan(
            level_name="AI Level",
            difficulty="hard",
            target_duration=30.0,
            object_budget=1000,
            style="ai_style",
            sync_intensity="high",
            sections=ai_sections
        )
        
        ai_response = AILevelPlanResponse(
            metadata={
                "planner_status": "success",
                "planner_result_plan": ai_level_plan,
                "level_plan": {},
                "sections": []
            },
            provider="ollama",
            model="mock-model"
        )
        mock_provider.generate_level_plan.return_value = ai_response

        # 3. Run generation
        config = {
            "audio_file": "tests/clicks.wav",
            "use_ai_planner": True,
            "output_name": "ai_test_level",
            "ollama_model": "mock-model"
        }
        
        result = generate_audio_synced_level_from_config(config)
        
        # 4. Verify results
        self.assertTrue(result["final_success"])
        self.assertEqual(result["planner_status"], "success")
        
        # Verify that spider mode was used in the final report
        report_path = Path("outputs/ai_test_level_report.json")
        self.assertTrue(report_path.exists())
        report = json.loads(report_path.read_text())
        print(f"DEBUG: Report section_plans: {report.get('section_plans')}")
        
        # Check if any section has 'spider' game mode
        found_spider = False
        for section in report.get("section_plans", []):
            if section.get("gameplay_mode") == "spider":
                found_spider = True
        
        self.assertTrue(found_spider, "AI-suggested 'spider' game mode was not found in final report sections")
        print("Test passed: AI sections were successfully applied to the pipeline.")

if __name__ == "__main__":
    unittest.main()
