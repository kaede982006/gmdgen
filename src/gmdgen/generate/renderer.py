# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from gmdgen.gd.plans import ObjectPlan, SectionPlan, TriggerPlan
from gmdgen.generate.role_mapping import choose_object_id_for_role
from gmdgen.render.object_materializer import materialize_object_roles
from gmdgen.render.trigger_materializer import materialize_trigger_roles


@dataclass(slots=True)
class ObjectRenderer:
    style_summary: dict[str, Any]
    safe_mode: bool = True
    seed: int = 0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def render_object_plan(self, plan: ObjectPlan, section: SectionPlan) -> ObjectPlan:
        plan.object_id = choose_object_id_for_role(
            plan.role,
            section_type=section.section_type,
            difficulty=section.difficulty_target,
            energy=section.density_target,
            style_summary=self.style_summary,
            safe_mode=self.safe_mode,
            rng=self._rng,
        )
        if section.section_type == "drop" and plan.role in {"safe_decoration", "ai_decoration", "visual_accent_target"}:
            plan.scale = max(plan.scale, 1.15)
        if section.section_type == "break":
            plan.scale = min(plan.scale, 1.0)
        return plan

    def render_gameplay_event(self, plan: ObjectPlan, section: SectionPlan) -> ObjectPlan:
        if "orb" in plan.role or "pad" in plan.role:
            return self.render_object_plan(plan, section)
        plan.role = "beat_orb" if section.section_type == "drop" else "ai_structure"
        return self.render_object_plan(plan, section)

    def render_decoration_cluster(self, x: float, y: float, section: SectionPlan, *, count: int = 3) -> list[ObjectPlan]:
        if section.section_type == "break":
            count = min(count, 1)
        if section.section_type == "drop":
            count = max(count, 4)
        plans = []
        for idx in range(count):
            role = "visual_accent_target" if section.section_type == "drop" and idx == 0 else "safe_decoration"
            plan = ObjectPlan(
                object_id="500",
                x=x + idx * 30,
                y=y + (idx % 2) * 30,
                role=role,
                safety_flags={"section_id": _section_index(section), "source": "renderer"},
            )
            plans.append(self.render_object_plan(plan, section))
        return plans

    def render_obstacle_pattern(self, x: float, y: float, section: SectionPlan, *, count: int = 2) -> list[ObjectPlan]:
        count = max(1, count if section.difficulty_target >= 0.45 else 1)
        return [
            self.render_object_plan(
                ObjectPlan(
                    object_id="8",
                    x=x + idx * 45,
                    y=y,
                    role="obstacle",
                    safety_flags={"section_id": _section_index(section), "source": "renderer"},
                ),
                section,
            )
            for idx in range(count)
        ]

    def render_orb_pad_sequence(self, x: float, y: float, section: SectionPlan, *, count: int = 3) -> list[ObjectPlan]:
        roles = ["beat_pad", "beat_orb", "beat_orb"] if section.section_type == "drop" else ["beat_pad", "ai_structure"]
        result = []
        for idx in range(max(1, count)):
            role = roles[idx % len(roles)]
            result.append(
                self.render_object_plan(
                    ObjectPlan(
                        object_id="35",
                        x=x + idx * 72,
                        y=y + (idx % 2) * 36,
                        role=role,
                        beat_aligned_time=section.start_time + idx * 0.25,
                        safety_flags={"section_id": _section_index(section), "source": "renderer"},
                    ),
                    section,
                )
            )
        return result

    def render_drop_accent(self, x: float, y: float, section: SectionPlan) -> tuple[list[ObjectPlan], list[TriggerPlan]]:
        objects = self.render_decoration_cluster(x, y, section, count=4)
        group_id = 1 + _section_index(section)
        for obj in objects:
            if group_id not in obj.group_ids:
                obj.group_ids.append(group_id)
        triggers = [
            TriggerPlan(
                trigger_type="pulse",
                object_id="1006",
                x=x,
                y=y + 90,
                target_group=group_id,
                duration=0.18,
                beat_aligned_time=section.start_time,
                properties={"color_channel": 1},
            )
        ]
        return objects, triggers

    def render_background_effect(self, x: float, y: float, section: SectionPlan) -> TriggerPlan:
        return TriggerPlan(
            trigger_type="color",
            object_id="29",
            x=x,
            y=y,
            target_group=None,
            duration=0.25 if section.section_type == "drop" else 0.5,
            beat_aligned_time=section.start_time,
            properties={"color_channel": 1},
        )


def render_plans_with_style(
    object_plans: list[ObjectPlan],
    section_plans: list[SectionPlan],
    *,
    style_summary: dict[str, Any] | None = None,
    safe_mode: bool = True,
    seed: int = 0,
) -> int:
    renderer = ObjectRenderer(style_summary or {}, safe_mode=safe_mode, seed=seed)
    
    # Run deterministic materialization for roles without hardcoded mappings
    materialize_object_roles(object_plans, style_summary=style_summary or {}, seed=seed)
    
    changed = 0
    for plan in object_plans:
        section = _section_for_plan(plan, section_plans)
        before = plan.object_id
        renderer.render_object_plan(plan, section)
        if plan.object_id != before:
            changed += 1
    return changed

def render_triggers_with_style(
    trigger_plans: list[TriggerPlan],
    *,
    safe_mode: bool = True,
    seed: int = 0,
) -> None:
    materialize_trigger_roles(trigger_plans, safe_mode=safe_mode, seed=seed)


def _section_for_plan(plan: ObjectPlan, section_plans: list[SectionPlan]) -> SectionPlan:
    section_id = plan.safety_flags.get("section_id") if isinstance(plan.safety_flags, dict) else None
    if isinstance(section_id, int) and 0 <= section_id < len(section_plans):
        return section_plans[section_id]
    for section in section_plans:
        if section.start_x <= plan.x <= section.end_x:
            return section
    return section_plans[0]


def _section_index(section: SectionPlan) -> int:
    try:
        return int(getattr(section, "section_id"))
    except Exception:
        return 0
