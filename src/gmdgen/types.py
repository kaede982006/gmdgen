# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 kaede982006
"""Hierarchical Structural Representation (HSR) types.

The AI is responsible for producing exactly one ``LevelPlan``; deterministic
expansion in ``gmdgen.generate.expand`` turns that plan into the full object
graph. This separation is the core of v2.3's call-count reduction.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


SectionKind = Literal["intro", "buildup", "drop", "break", "climax", "outro"]
GameMode = Literal["cube", "ship", "ball", "ufo", "wave", "robot", "spider"]
DifficultyTier = Literal["easy", "medium", "hard"]

VALID_SECTION_KINDS: tuple[SectionKind, ...] = (
    "intro", "buildup", "drop", "break", "climax", "outro",
)
VALID_GAME_MODES: tuple[GameMode, ...] = (
    "cube", "ship", "ball", "ufo", "wave", "robot", "spider",
)
VALID_DIFFICULTIES: tuple[DifficultyTier, ...] = ("easy", "medium", "hard")


@dataclass(slots=True)
class Transitions:
    camera: str = "default"
    color: str = "default"
    gravity: float = 1.0
    speed: float = 1.0


@dataclass(slots=True)
class LevelMeta:
    name: str = "untitled"
    song_id: str = ""
    target_difficulty: DifficultyTier = "medium"
    target_length_seconds: float = 60.0


@dataclass(slots=True)
class Section:
    id: str
    kind: SectionKind
    length_beats: int
    bpm: int
    mode: GameMode
    intensity: float = 0.5
    pattern_refs: list[str] = field(default_factory=list)
    transitions: Transitions = field(default_factory=Transitions)

    def __post_init__(self) -> None:
        if not (4 <= self.length_beats <= 64):
            raise ValueError(f"length_beats must be in [4, 64], got {self.length_beats}")
        if not (60 <= self.bpm <= 300):
            raise ValueError(f"bpm must be in [60, 300], got {self.bpm}")
        if not (0.0 <= self.intensity <= 1.0):
            raise ValueError(f"intensity must be in [0.0, 1.0], got {self.intensity}")
        if self.kind not in VALID_SECTION_KINDS:
            raise ValueError(f"invalid section kind: {self.kind}")
        if self.mode not in VALID_GAME_MODES:
            raise ValueError(f"invalid game mode: {self.mode}")


@dataclass(slots=True)
class LevelPlan:
    meta: LevelMeta
    sections: list[Section]

    def __post_init__(self) -> None:
        if not self.sections:
            raise ValueError("LevelPlan must contain at least one section")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LevelPlan":
        meta_raw = payload.get("meta", {}) or {}
        meta = LevelMeta(**meta_raw)
        sections: list[Section] = []
        for raw in payload.get("sections", []) or []:
            tr = Transitions(**(raw.get("transitions") or {}))
            sections.append(Section(
                id=str(raw["id"]),
                kind=raw["kind"],
                length_beats=int(raw["length_beats"]),
                bpm=int(raw["bpm"]),
                mode=raw["mode"],
                intensity=float(raw.get("intensity", 0.5)),
                pattern_refs=list(raw.get("pattern_refs") or []),
                transitions=tr,
            ))
        return cls(meta=meta, sections=sections)


# JSON schema used to constrain Ollama output.
LEVEL_PLAN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["meta", "sections"],
    "properties": {
        "meta": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "song_id": {"type": "string"},
                "target_difficulty": {"enum": list(VALID_DIFFICULTIES)},
                "target_length_seconds": {"type": "number"},
            },
        },
        "sections": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["id", "kind", "length_beats", "bpm", "mode"],
                "properties": {
                    "id": {"type": "string"},
                    "kind": {"enum": list(VALID_SECTION_KINDS)},
                    "length_beats": {"type": "integer", "minimum": 4, "maximum": 64},
                    "bpm": {"type": "integer", "minimum": 60, "maximum": 300},
                    "mode": {"enum": list(VALID_GAME_MODES)},
                    "intensity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "pattern_refs": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}
