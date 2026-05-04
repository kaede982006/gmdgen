from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class FeatureRecord:
    example_id: str = ""
    audio_features: dict[str, Any] = field(default_factory=dict)
    section_features: dict[str, Any] = field(default_factory=dict)
    rhythm_features: dict[str, Any] = field(default_factory=dict)
    motif_features: dict[str, Any] = field(default_factory=dict)
    plan_features: dict[str, Any] = field(default_factory=dict)
    renderer_features: dict[str, Any] = field(default_factory=dict)
    repair_features: dict[str, Any] = field(default_factory=dict)
    quality_features: dict[str, Any] = field(default_factory=dict)
    geode_features: dict[str, Any] = field(default_factory=dict)
    user_feedback_features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "audio_features": dict(self.audio_features),
            "section_features": dict(self.section_features),
            "rhythm_features": dict(self.rhythm_features),
            "motif_features": dict(self.motif_features),
            "plan_features": dict(self.plan_features),
            "renderer_features": dict(self.renderer_features),
            "repair_features": dict(self.repair_features),
            "quality_features": dict(self.quality_features),
            "geode_features": dict(self.geode_features),
            "user_feedback_features": dict(self.user_feedback_features),
        }


@dataclass(slots=True)
class LabelRecord:
    example_id: str = ""
    quality_gate_passed: bool = False
    user_rating: int = 0
    user_tags: list[str] = field(default_factory=list)
    good_sync: bool = False
    bad_sync: bool = False
    good_drop: bool = False
    bad_drop: bool = False
    too_empty: bool = False
    too_dense: bool = False
    boring: bool = False
    not_gd_like: bool = False
    editor_problem: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "quality_gate_passed": self.quality_gate_passed,
            "user_rating": self.user_rating,
            "user_tags": list(self.user_tags),
            "good_sync": self.good_sync,
            "bad_sync": self.bad_sync,
            "good_drop": self.good_drop,
            "bad_drop": self.bad_drop,
            "too_empty": self.too_empty,
            "too_dense": self.too_dense,
            "boring": self.boring,
            "not_gd_like": self.not_gd_like,
            "editor_problem": self.editor_problem,
        }


def build_feature_record_from_generation(example_id: str, report: dict[str, Any]) -> FeatureRecord:
    record = FeatureRecord(example_id=example_id)
    record.quality_features = dict(report.get("metrics", {}))
    return record


def build_label_record_from_feedback(example_id: str, feedback: dict[str, Any]) -> LabelRecord:
    return LabelRecord(
        example_id=example_id,
        user_rating=int(feedback.get("rating", 0)),
        quality_gate_passed=bool(feedback.get("quality_gate_passed", False)),
    )


def split_train_val_test(dataset_dir: Path) -> dict[str, list[str]]:
    return {"train": [], "val": [], "test": []}


def validate_ml_dataset(dataset_dir: Path) -> list[str]:
    return []
