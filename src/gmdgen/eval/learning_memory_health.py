from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class LearningMemoryHealthReport:
    total_examples: int = 0
    high_quality_examples: int = 0
    accepted_examples: int = 0
    unreviewed_examples: int = 0
    low_quality_examples: int = 0
    rejected_examples: int = 0
    user_favorites: int = 0
    failure_patterns: int = 0
    preference_pairs: int = 0
    examples_used_in_context: int = 0
    examples_excluded_from_context: int = 0
    exclusion_reasons: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_examples": self.total_examples,
            "high_quality_examples": self.high_quality_examples,
            "accepted_examples": self.accepted_examples,
            "unreviewed_examples": self.unreviewed_examples,
            "low_quality_examples": self.low_quality_examples,
            "rejected_examples": self.rejected_examples,
            "user_favorites": self.user_favorites,
            "failure_patterns": self.failure_patterns,
            "preference_pairs": self.preference_pairs,
            "examples_used_in_context": self.examples_used_in_context,
            "examples_excluded_from_context": self.examples_excluded_from_context,
            "exclusion_reasons": dict(self.exclusion_reasons),
        }


def analyze_learning_memory_health(store_dir: Path) -> LearningMemoryHealthReport:
    report = LearningMemoryHealthReport()
    # Mock analysis for now
    return report


def build_success_memory_context(store_dir: Path) -> list[dict[str, Any]]:
    return []


def build_failure_pattern_context(store_dir: Path) -> list[dict[str, Any]]:
    return []


def build_preference_pairs_from_feedback(store_dir: Path) -> list[dict[str, Any]]:
    return []
