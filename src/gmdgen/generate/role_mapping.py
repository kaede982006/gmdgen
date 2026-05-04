# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import random
from collections import Counter
from typing import Any

from gmdgen.gd.plans import ObjectPlan, SectionPlan
from gmdgen.representation.object_classifier import ObjectClass, classify


SAFE_ROLE_OBJECT_POOLS: dict[str, tuple[str, ...]] = {
    "ground_or_structure": ("1", "2", "3", "4"),
    "structure": ("1", "2", "3", "4"),
    "ai_structure": ("1", "2", "3", "4"),
    "section_transition_structure": ("1", "2", "3", "4", "6"),
    "structure_accent": ("8", "39"),
    "obstacle": ("8", "39"),
    "beat_orb": ("36", "84", "141", "1022"),
    "ai_orb": ("36", "84", "141", "1022"),
    "orb": ("36", "84", "141", "1022"),
    "beat_pad": ("35", "140"),
    "ai_pad": ("35", "140"),
    "jump_pad": ("35", "140"),
    "visual_accent_target": ("500", "501", "503"),
    "decoration": ("500", "501", "503", "504"),
    "safe_decoration": ("500", "501", "503", "504"),
    "ai_decoration": ("500", "501", "503", "504"),
    "background_effect": ("500", "501", "503"),
}


ADVANCED_ROLE_OBJECT_POOLS: dict[str, tuple[str, ...]] = {
    **SAFE_ROLE_OBJECT_POOLS,
    "decoration": ("500", "501", "503", "504", "505", "579"),
    "safe_decoration": ("500", "501", "503", "504", "505", "579"),
    "ai_decoration": ("500", "501", "503", "504", "505", "579"),
    "structure_accent": ("8", "39", "6"),
}


def build_role_object_pool(style_summary: dict[str, Any] | None = None, safe_mode: bool = True) -> dict[str, list[str]]:
    base = SAFE_ROLE_OBJECT_POOLS if safe_mode else ADVANCED_ROLE_OBJECT_POOLS
    pool = {role: list(ids) for role, ids in base.items()}
    ids_by_class = (style_summary or {}).get("ids_by_class", {})
    if isinstance(ids_by_class, dict):
        for object_id in ids_by_class.get(ObjectClass.DECORATION.value, [])[:12]:
            if _safe_object_id(str(object_id), safe_mode):
                pool.setdefault("decoration", []).append(str(object_id))
                pool.setdefault("ai_decoration", []).append(str(object_id))
        for object_id in ids_by_class.get(ObjectClass.STRUCTURE.value, [])[:12]:
            if _safe_object_id(str(object_id), safe_mode):
                pool.setdefault("structure", []).append(str(object_id))
                pool.setdefault("ai_structure", []).append(str(object_id))
    learned_distribution = (style_summary or {}).get("learned_object_distribution", {})
    if isinstance(learned_distribution, dict):
        for object_id, _count in sorted(
            learned_distribution.items(),
            key=lambda item: int(item[1]) if str(item[1]).isdigit() else 0,
            reverse=True,
        )[:24]:
            object_id = str(object_id)
            if not _safe_object_id(object_id, safe_mode):
                continue
            try:
                object_class = classify(object_id)
            except Exception:
                continue
            if object_class == ObjectClass.DECORATION:
                pool.setdefault("decoration", []).append(object_id)
                pool.setdefault("ai_decoration", []).append(object_id)
                pool.setdefault("visual_accent_target", []).append(object_id)
            elif object_class == ObjectClass.STRUCTURE:
                pool.setdefault("structure", []).append(object_id)
                pool.setdefault("ai_structure", []).append(object_id)
                pool.setdefault("ground_or_structure", []).append(object_id)
            elif object_id in {"35", "140"}:
                pool.setdefault("jump_pad", []).append(object_id)
                pool.setdefault("beat_pad", []).append(object_id)
            elif object_id in {"36", "84", "141", "1022"}:
                pool.setdefault("orb", []).append(object_id)
                pool.setdefault("beat_orb", []).append(object_id)
    return {role: _dedupe(ids) for role, ids in pool.items()}


def choose_object_id_for_role(
    role: str,
    *,
    section_type: str = "normal",
    difficulty: float = 0.5,
    energy: float = 0.5,
    style_summary: dict[str, Any] | None = None,
    safe_mode: bool = True,
    rng: random.Random | None = None,
) -> str:
    rng = rng or random.Random(0)
    pool = build_role_object_pool(style_summary, safe_mode)
    ids = pool.get(role) or pool.get(_fallback_role(role), ["1"])
    if section_type == "drop" and role in {"decoration", "safe_decoration", "ai_decoration", "visual_accent_target"}:
        ids = list(ids) + list(ids[-2:])
    if role in {"structure_accent", "obstacle"} and difficulty < 0.45:
        ids = ["1"]
    if not ids:
        return "1"
    index = min(len(ids) - 1, int(rng.random() * len(ids)))
    return str(ids[index])


def diversify_object_ids(
    object_plans: list[ObjectPlan],
    *,
    section_plans: list[SectionPlan],
    style_summary: dict[str, Any] | None = None,
    difficulty: float = 0.5,
    safe_mode: bool = True,
    seed: int = 0,
) -> int:
    rng = random.Random(seed)
    changed = 0
    previous_by_role: dict[str, str] = {}
    for idx, plan in enumerate(object_plans):
        role = str(plan.role)
        if role == "speed_portal":
            continue
        section = _section_for_plan(plan, section_plans)
        chosen = choose_object_id_for_role(
            role,
            section_type=section.section_type if section else "normal",
            difficulty=difficulty,
            energy=section.density_target if section else 0.5,
            style_summary=style_summary,
            safe_mode=safe_mode,
            rng=rng,
        )
        if role in {"beat_orb", "ai_orb", "orb", "beat_pad", "ai_pad", "jump_pad"}:
            should_force = True
        else:
            should_force = str(plan.object_id) not in build_role_object_pool(style_summary, safe_mode).get(role, [str(plan.object_id)])
        if previous_by_role.get(role) == chosen:
            alternatives = [item for item in build_role_object_pool(style_summary, safe_mode).get(role, [chosen]) if item != chosen]
            if alternatives:
                chosen = alternatives[idx % len(alternatives)]
        if should_force and str(plan.object_id) != chosen:
            plan.object_id = chosen
            changed += 1
        previous_by_role[role] = str(plan.object_id)
    return changed


def object_diversity_for_plans(object_plans: list[ObjectPlan]) -> float:
    if not object_plans:
        return 0.0
    ids = [str(plan.object_id) for plan in object_plans]
    return len(set(ids)) / len(ids)


def _fallback_role(role: str) -> str:
    if "orb" in role:
        return "beat_orb"
    if "pad" in role:
        return "beat_pad"
    if "decor" in role or "motif" in role:
        return "decoration"
    if "accent" in role:
        return "structure_accent"
    return "structure"


def _safe_object_id(object_id: str, safe_mode: bool) -> bool:
    if not safe_mode:
        return True
    try:
        cls = classify(object_id)
    except Exception:
        return False
    return cls in {ObjectClass.STRUCTURE, ObjectClass.DECORATION, ObjectClass.SPECIAL}


def _section_for_plan(plan: ObjectPlan, section_plans: list[SectionPlan]) -> SectionPlan | None:
    section_id = plan.safety_flags.get("section_id") if isinstance(plan.safety_flags, dict) else None
    if isinstance(section_id, int) and 0 <= section_id < len(section_plans):
        return section_plans[section_id]
    for section in section_plans:
        if section.start_x <= plan.x <= section.end_x:
            return section
    return None


def _dedupe(values: list[str]) -> list[str]:
    counts = Counter(values)
    return list(counts.keys())
