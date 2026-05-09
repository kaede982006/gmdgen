# SPDX-License-Identifier: GPL-3.0-or-later
import json
import unittest
from unittest.mock import MagicMock, patch

import torch
from gmdgen.ai.planner import parse_ollama_section_plan
from gmdgen.utils.device import get_device_info, get_best_device, apply_device_info_to_report
from gmdgen.gd.plans import ValidationReport


class TestPlannerHardening(unittest.TestCase):
    def test_missing_level_plan_and_sections(self):
        payload = {}
        result = parse_ollama_section_plan(payload)
        self.assertFalse(result.valid)
        self.assertIn("$.level_plan", result.missing_required_fields)
        self.assertIn("$.sections", result.missing_required_fields)
        self.assertEqual(
            result.schema_error_message,
            "Both top-level level_plan and non-empty top-level sections are required."
        )

    def test_missing_level_plan_only(self):
        payload = {"sections": [{"section_id": "s1", "time_start": 0, "time_end": 10, "game_mode": "cube", "speed": "1x", "density": 0.5, "primary_pattern": "test"}]}
        result = parse_ollama_section_plan(payload)
        self.assertFalse(result.valid)
        self.assertIn("$.level_plan", result.missing_required_fields)
        self.assertEqual(result.schema_error_message, "Level_plan is missing.")

    def test_empty_sections(self):
        payload = {
            "level_plan": {
                "level_name": "test", "difficulty": "normal", "target_duration": 30,
                "object_budget": 500, "style": "modern", "sync_intensity": "medium"
            },
            "sections": []
        }
        result = parse_ollama_section_plan(payload)
        self.assertFalse(result.valid)
        self.assertIn("$.sections", result.empty_required_fields)
        self.assertEqual(result.schema_error_message, "Sections must be a non-empty array.")

    def test_wrong_location_sections(self):
        payload = {
            "level_plan": {
                "level_name": "test", "difficulty": "normal", "target_duration": 30,
                "object_budget": 500, "style": "modern", "sync_intensity": "medium",
                "sections": [{"section_id": "s1", "time_start": 0, "time_end": 10, "game_mode": "cube", "speed": "1x", "density": 0.5, "primary_pattern": "test"}]
            }
        }
        result = parse_ollama_section_plan(payload)
        self.assertTrue(result.valid)
        self.assertIn("moved_$.level_plan.sections_to_$.sections", result.normalized_shape_repairs)
        self.assertIn("$.level_plan.sections", result.wrong_location_fields)

    def test_alias_normalization(self):
        payload = {
            "plan": { # alias for level_plan
                "name": "test", # alias for level_name
                "difficulty": "hard-demon", # normalize to demon
                "duration": 60, # alias for target_duration
                "object_budget": 500, "style": "modern", "sync_intensity": "intense" # normalize to high
            },
            "section_list": [ # alias for sections
                {
                    "section_id": "s1", "time_start": 0, "time_end": 10, 
                    "game_mode": "ship gameplay", # normalize to ship
                    "speed": "2 speed", # normalize to 2x
                    "target_density": 0.5, # alias for density
                    "primary_pattern": "test"
                }
            ]
        }
        result = parse_ollama_section_plan(payload)
        self.assertTrue(result.valid)
        self.assertEqual(result.plan.level_name, "test")
        self.assertEqual(result.plan.difficulty, "demon")
        self.assertEqual(result.plan.target_duration, 60.0)
        self.assertEqual(result.plan.sync_intensity, "high")
        self.assertEqual(result.plan.sections[0].game_mode, "ship")
        self.assertEqual(result.plan.sections[0].speed, "2x")
        self.assertEqual(result.plan.sections[0].density, 0.5)

    def test_forbidden_fields(self):
        payload = {
            "level_plan": {
                "level_name": "test", "difficulty": "normal", "target_duration": 30,
                "object_budget": 500, "style": "modern", "sync_intensity": "medium"
            },
            "sections": [
                {
                    "section_id": "s1", "time_start": 0, "time_end": 10, "game_mode": "cube", "speed": "1x", "density": 0.5, "primary_pattern": "test",
                    "object_plans": [{"id": 1}] # Forbidden
                }
            ],
            "score": 100 # Forbidden
        }
        result = parse_ollama_section_plan(payload)
        self.assertFalse(result.valid)
        self.assertIn("object_plans", result.forbidden_fields)
        self.assertIn("score", result.forbidden_fields)


class TestGPUDetection(unittest.TestCase):
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090")
    def test_cuda_detection(self, mock_name, mock_is_available):
        from gmdgen.utils.device import get_device_info
        get_device_info.cache_clear()
        info = get_device_info()
        self.assertEqual(info.compute_device, "cuda")
        self.assertTrue(info.gpu_available)
        self.assertEqual(info.gpu_name, "NVIDIA RTX 4090")
        self.assertEqual(info.gpu_backend, "cuda")

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_mps_detection(self, mock_mps, mock_cuda):
        from gmdgen.utils.device import get_device_info
        get_device_info.cache_clear()
        info = get_device_info()
        self.assertEqual(info.compute_device, "mps")
        self.assertTrue(info.gpu_available)
        self.assertEqual(info.gpu_backend, "mps")

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_cpu_fallback(self, mock_mps, mock_cuda):
        from gmdgen.utils.device import get_device_info
        get_device_info.cache_clear()
        info = get_device_info()
        self.assertEqual(info.compute_device, "cpu")
        self.assertFalse(info.gpu_available)
        self.assertEqual(info.fallback_reason, "no_gpu_detected")

    def test_apply_to_report(self):
        report = ValidationReport()
        apply_device_info_to_report(report)
        self.assertEqual(report.compute_device, get_best_device().type if torch else "cpu")
        self.assertEqual(report.torch_available, torch is not None)


if __name__ == "__main__":
    unittest.main()
