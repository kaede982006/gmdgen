# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import asdict

from gmdgen.generate.ir import GMDObjectIR, LevelIR, LevelPlan, SectionIR, SectionPlan, TriggerIR


def section_plan_to_ir(plan: SectionPlan) -> SectionIR:
    """Decode a symbolic planner section into local IR.

    The planner can choose families and symbols, but concrete Geometry Dash ids
    are introduced locally. This keeps the serializer and validators as the
    source of truth for the final .gmd output.
    """
    base_y = 105.0
    span = max(1.0, plan.time_end - plan.time_start)
    density_count = max(1, int(round(2 + plan.density * 6)))
    group_symbols = list(plan.group_symbols)
    color_symbols = list(plan.color_symbols)
    primary_group = group_symbols[0] if group_symbols else None
    primary_color = color_symbols[0] if color_symbols else None
    objects: list[GMDObjectIR] = []
    for index in range(density_count):
        family = plan.allowed_object_families[index % max(1, len(plan.allowed_object_families))]
        object_id = _object_id_for_family(family)
        x = (plan.time_start + span * (index / max(1, density_count - 1))) * 30.0
        y = base_y + (index % 3) * 30.0
        objects.append(
            GMDObjectIR(
                object_id=object_id,
                x=x,
                y=y,
                role="gameplay" if family in {"spike", "orb", "pad"} else "structural",
                group_symbols=[primary_group] if primary_group is not None else [],
                color_symbol=primary_color,
                properties={"family": family, "source": "section_plan_decoder"},
            )
        )

    triggers: list[TriggerIR] = []
    if plan.trigger_budget > 0 and primary_group is not None:
        triggers.append(
            TriggerIR(
                trigger_type="pulse",
                x=plan.time_start * 30.0,
                y=255.0,
                target_group_symbol=primary_group,
                color_symbol=primary_color,
                duration=0.18,
                properties={"source": "section_plan_decoder"},
            )
        )

    return SectionIR(
        section_id=plan.section_id,
        time_start=plan.time_start,
        time_end=plan.time_end,
        game_mode=plan.game_mode,
        speed=plan.speed,
        density=plan.density,
        objects=objects,
        triggers=triggers,
        group_symbols=group_symbols,
        color_symbols=color_symbols,
        source_plan=asdict(plan),
    )


def level_plan_to_ir(plan: LevelPlan) -> LevelIR:
    return LevelIR(
        level_name=plan.level_name,
        difficulty=plan.difficulty,
        target_duration=plan.target_duration,
        object_budget=plan.object_budget,
        style=plan.style,
        sync_intensity=plan.sync_intensity,
        sections=[section_plan_to_ir(section) for section in plan.sections],
    )


def _object_id_for_family(family: str) -> str:
    return {
        "block": "1",
        "spike": "8",
        "orb": "36",
        "pad": "35",
        "decoration": "211",
    }.get(family, "1")
