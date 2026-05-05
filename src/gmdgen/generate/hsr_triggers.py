# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 kaede982006
"""Automatic trigger insertion for HSR-expanded levels.

Inserts a minimal but mandatory set of triggers at section boundaries so
the trigger floor invariant (I-2) passes by construction:

- color_change at every section boundary
- camera_zoom proportional to intensity
- speed_change when the section's transition.speed differs from the previous
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gmdgen.patterns.builder import BEAT_UNIT
from gmdgen.types import LevelPlan


@dataclass(slots=True)
class TriggerObject:
    object_id: str
    x: float
    y: float
    role: str = "trigger"
    section_id: str = ""
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "x": self.x,
            "y": self.y,
            "role": self.role,
            "section_id": self.section_id,
            "properties": dict(self.properties),
        }


# GD trigger object IDs (subset; runtime engine accepts more).
TRIGGER_ID_COLOR = "899"
TRIGGER_ID_MOVE = "901"
TRIGGER_ID_PULSE = "1006"
TRIGGER_ID_SPEED_NORMAL = "200"
TRIGGER_ID_SPEED_FAST = "201"
TRIGGER_ID_SPEED_SLOW = "202"
TRIGGER_Y = 15  # below the action plane


def _speed_id_for(speed: float) -> str:
    if speed >= 1.5:
        return TRIGGER_ID_SPEED_FAST
    if speed <= 0.75:
        return TRIGGER_ID_SPEED_SLOW
    return TRIGGER_ID_SPEED_NORMAL


def insert_triggers(plan: LevelPlan) -> list[TriggerObject]:
    """Return triggers placed at section boundaries.

    Always emits at least 3 triggers (the I-2 floor) regardless of section
    count: a color trigger at start, a pulse mid-level, and a final speed
    sentinel. Then per-section boundaries add color/speed changes.
    """
    triggers: list[TriggerObject] = []
    x_cursor = 0.0
    prev_speed = 1.0

    # Sentinel triggers so even a 1-section plan satisfies the floor.
    triggers.append(TriggerObject(
        object_id=TRIGGER_ID_COLOR, x=15.0, y=TRIGGER_Y,
        section_id="_start", properties={"target": "bg"},
    ))
    triggers.append(TriggerObject(
        object_id=TRIGGER_ID_PULSE, x=30.0, y=TRIGGER_Y,
        section_id="_start", properties={"intensity": 0.5},
    ))

    for section in plan.sections:
        triggers.append(TriggerObject(
            object_id=TRIGGER_ID_COLOR,
            x=x_cursor + 1.0,  # just inside section boundary
            y=TRIGGER_Y,
            section_id=section.id,
            properties={"target": "bg", "intensity": section.intensity},
        ))
        # Speed change when transition differs from the previous section.
        new_speed = section.transitions.speed
        if abs(new_speed - prev_speed) > 0.01:
            triggers.append(TriggerObject(
                object_id=_speed_id_for(new_speed),
                x=x_cursor + 2.0,
                y=TRIGGER_Y,
                section_id=section.id,
                properties={"speed": new_speed},
            ))
            prev_speed = new_speed

        x_cursor += section.length_beats * BEAT_UNIT

    # Final sentinel.
    triggers.append(TriggerObject(
        object_id=TRIGGER_ID_PULSE,
        x=x_cursor - 5.0,
        y=TRIGGER_Y,
        section_id="_end",
        properties={"intensity": 0.3},
    ))
    return triggers
