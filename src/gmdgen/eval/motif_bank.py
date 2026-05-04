# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class MotifQualityScore:
    density_score: float = 0.0
    structure_score: float = 0.0
    trigger_validity_score: float = 0.0
    style_consistency_score: float = 0.0
    gameplay_usefulness_score: float = 0.0
    section_relevance_score: float = 0.0
    reusability_score: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.density_score
            + self.structure_score
            + self.trigger_validity_score
            + self.style_consistency_score
            + self.gameplay_usefulness_score
            + self.section_relevance_score
            + self.reusability_score
        ) / 7.0

    def to_dict(self) -> dict[str, float]:
        return {
            "density_score": self.density_score,
            "structure_score": self.structure_score,
            "trigger_validity_score": self.trigger_validity_score,
            "style_consistency_score": self.style_consistency_score,
            "gameplay_usefulness_score": self.gameplay_usefulness_score,
            "section_relevance_score": self.section_relevance_score,
            "reusability_score": self.reusability_score,
            "total": self.total,
        }


@dataclass(slots=True)
class MotifBank:
    motif_id: str = ""
    source_level: str = ""
    source_style: str = ""
    start_x: float = 0.0
    end_x: float = 0.0
    length_x: float = 0.0
    section_type_hint: str = ""
    difficulty_hint: str = ""
    gameplay_mode_hint: str = ""
    speed_hint: str = ""
    density: float = 0.0
    roles: list[str] = field(default_factory=list)
    object_ids: list[str] = field(default_factory=list)
    trigger_types: list[str] = field(default_factory=list)
    group_usage: list[int] = field(default_factory=list)
    rhythm_hint: str = ""
    style_tags: list[str] = field(default_factory=list)
    quality_score: MotifQualityScore = field(default_factory=MotifQualityScore)
    compact_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "motif_id": self.motif_id,
            "source_level": self.source_level,
            "source_style": self.source_style,
            "start_x": self.start_x,
            "end_x": self.end_x,
            "length_x": self.length_x,
            "section_type_hint": self.section_type_hint,
            "difficulty_hint": self.difficulty_hint,
            "gameplay_mode_hint": self.gameplay_mode_hint,
            "speed_hint": self.speed_hint,
            "density": self.density,
            "roles": list(self.roles),
            "object_ids": list(self.object_ids),
            "trigger_types": list(self.trigger_types),
            "group_usage": list(self.group_usage),
            "rhythm_hint": self.rhythm_hint,
            "style_tags": list(self.style_tags),
            "quality_score": self.quality_score.to_dict(),
            "compact_summary": self.compact_summary,
        }


def build_motif_bank_from_dataset(dataset_dir: Path) -> list[MotifBank]:
    return []


def score_motifs(motifs: list[MotifBank]) -> None:
    pass


def retrieve_motifs_for_section(motifs: list[MotifBank], section_type: str, count: int = 3) -> list[MotifBank]:
    return []


def summarize_motifs_for_prompt(motifs: list[MotifBank]) -> str:
    return ""


def validate_motif_bank_quality(motifs: list[MotifBank]) -> dict[str, Any]:
    return {}
