import pytest
from gmdgen.diagnostics.string_sanitizer import MetadataSanitizer
from gmdgen.generate.quality_gate import evaluate_quality_gate, QualityGateThresholds

def test_sanitizer_blocks_prompt_leak():
    sanitizer = MetadataSanitizer()
    bad_string = "Here is the JSON plan you requested: [object]"
    assert "JSON plan" not in sanitizer.sanitize_description(bad_string, fallback="fallback")

def test_sanitizer_blocks_garbage_json():
    sanitizer = MetadataSanitizer()
    bad_string = "{\"section_id\": 1, \"start_time\": 0.0}"
    assert "{" not in sanitizer.sanitize_description(bad_string, fallback="fallback")

def test_quality_gate_rejects_empty_drop():
    report = {
        "score": 0.5,
        "final_object_count": 20,
        "removed_object_ratio": 0.1,
        "empty_important_sections": ["drop"],
        "metrics": {
            "density_alignment_score": 0.8,
            "drop_impact_score": 0.1,  # Fails threshold
        }
    }
    thresholds = QualityGateThresholds(min_drop_impact=0.35)
    result = evaluate_quality_gate(report, thresholds)
    assert not result.passed
    assert any("drop_impact_below_threshold" in f for f in result.failures)

def test_quality_gate_rejects_high_repair_loss():
    report = {
        "score": 0.8,
        "final_object_count": 100,
        "removed_object_ratio": 0.9, # 90% objects repaired/removed
        "metrics": {
            "density_alignment_score": 0.8,
            "drop_impact_score": 0.8,
        }
    }
    thresholds = QualityGateThresholds(max_repair_loss=0.5)
    result = evaluate_quality_gate(report, thresholds)
    assert not result.passed
    assert any("repair_loss_above_threshold" in f for f in result.failures)
