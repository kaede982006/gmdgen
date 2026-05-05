# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 kaede982006
"""Deterministic expander: ``LevelPlan`` -> object graph.

This is the heart of v2.3's call-count reduction. The AI produces only a
``LevelPlan`` (~few hundred tokens); ``expand_plan`` walks the plan, picks
patterns from ``patterns_index.json``, and emits absolute-coordinate
gameplay/decoration objects. No AI calls are made here; same (plan, seed)
yields the same output.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from gmdgen.observability.log import logged
from gmdgen.patterns.builder import BEAT_UNIT, load_index, pick_pattern
from gmdgen.types import LevelPlan, Section


@dataclass(slots=True)
class ExpandedObject:
    object_id: str
    x: float
    y: float
    role: str = "structural"
    section_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "x": self.x,
            "y": self.y,
            "role": self.role,
            "section_id": self.section_id,
        }


@dataclass(slots=True)
class ExpansionResult:
    objects: list[ExpandedObject] = field(default_factory=list)
    section_object_counts: dict[str, int] = field(default_factory=dict)
    used_pattern_ids: list[str] = field(default_factory=list)

    @property
    def total_objects(self) -> int:
        return len(self.objects)


def _instantiate(
    pattern: dict[str, Any],
    *,
    x_cursor: float,
    section: Section,
    intensity_factor: float,
) -> list[ExpandedObject]:
    """Translate beat-relative pattern objects into absolute coordinates."""
    out: list[ExpandedObject] = []
    for raw in pattern.get("objects", []):
        obj_id = str(raw.get("id", "1"))
        x_beat = float(raw.get("x_beat", 0.0))
        y = float(raw.get("y", 105))
        role = str(raw.get("role", "structural"))

        # Intensity modulation: high intensity adds tiny vertical jitter to
        # decorations only — gameplay objects keep exact y.
        if role == "decoration":
            y_jitter = (intensity_factor - 0.5) * 12.0
            y += y_jitter

        x = x_cursor + x_beat * BEAT_UNIT
        out.append(ExpandedObject(
            object_id=obj_id,
            x=x,
            y=y,
            role=role,
            section_id=section.id,
        ))
    return out


@logged(phase="layout", step="expand_plan")
def expand_plan(plan: LevelPlan, *, seed: int = 0) -> ExpansionResult:
    """Materialize a ``LevelPlan`` into a deterministic object graph."""
    rng = random.Random(seed)
    index = load_index()
    result = ExpansionResult()
    x_cursor = 0.0

    for section in plan.sections:
        difficulty = "easy"
        # Map intensity to difficulty tier deterministically.
        if section.intensity >= 0.7:
            difficulty = "hard"
        elif section.intensity >= 0.35:
            difficulty = "medium"

        pattern = pick_pattern(
            mode=section.mode,
            difficulty=difficulty,
            rng=rng,
            index=index,
        )
        objs = _instantiate(
            pattern,
            x_cursor=x_cursor,
            section=section,
            intensity_factor=section.intensity,
        )
        result.objects.extend(objs)
        result.section_object_counts[section.id] = len(objs)
        result.used_pattern_ids.append(pattern["id"])
        x_cursor += section.length_beats * BEAT_UNIT

    # x is monotone by construction (we only advance the cursor).
    return result
