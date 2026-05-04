# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from gmdgen.gd.plans import ObjectPlan, SectionPlan
from gmdgen.gd.time_mapping import SpeedState
from gmdgen.generate.role_mapping import (
    build_role_object_pool,
    choose_object_id_for_role,
    diversify_object_ids,
)


def _section(section_type: str = "normal") -> SectionPlan:
    return SectionPlan(0, 1, 0, 600, section_type, "cube", SpeedState.NORMAL, 0.5, 0.5, 0.5, 0.5)


def test_jump_pad_maps_to_pad_object_pool() -> None:
    pool = build_role_object_pool({}, True)

    assert choose_object_id_for_role("jump_pad", safe_mode=True) in pool["jump_pad"]


def test_orb_maps_to_orb_object_pool() -> None:
    pool = build_role_object_pool({}, True)

    assert choose_object_id_for_role("orb", safe_mode=True) in pool["orb"]


def test_decoration_mapping_uses_style_summary() -> None:
    style = {"ids_by_class": {"decoration": ["777"]}}
    pool = build_role_object_pool(style, safe_mode=False)

    assert "777" in pool["decoration"]


def test_object_id_diversification_reduces_repetition() -> None:
    plans = [ObjectPlan("9999", idx * 60, 90, "beat_orb") for idx in range(6)]

    changed = diversify_object_ids(
        plans,
        section_plans=[_section()],
        safe_mode=True,
        seed=3,
    )

    assert changed == 6
    assert len({plan.object_id for plan in plans}) > 1


def test_safe_mode_uses_safe_object_ids_only() -> None:
    pool = build_role_object_pool({"ids_by_class": {"decoration": ["1006", "500"]}}, safe_mode=True)

    assert "1006" not in pool["decoration"]
    assert "500" in pool["decoration"]


def test_role_mapping_uses_learned_object_distribution() -> None:
    style = {"learned_object_distribution": {"500": 8, "1": 4, "1006": 2}}
    pool = build_role_object_pool(style, safe_mode=True)

    assert "500" in pool["decoration"]
    assert "1" in pool["structure"]
    assert "1006" not in pool["decoration"]
