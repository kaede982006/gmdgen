from __future__ import annotations

from gmdgen.eval.ml_dataset import FeatureRecord, LabelRecord, build_feature_record_from_generation, build_label_record_from_feedback


def test_feature_record_created_from_generation_report() -> None:
    report = {"metrics": {"playability_safety_score": 0.9}}
    record = build_feature_record_from_generation("test1", report)
    assert record.example_id == "test1"
    assert record.quality_features["playability_safety_score"] == 0.9

def test_label_record_created_from_feedback() -> None:
    feedback = {"rating": 5, "quality_gate_passed": True}
    label = build_label_record_from_feedback("test1", feedback)
    assert label.user_rating == 5
    assert label.quality_gate_passed is True
