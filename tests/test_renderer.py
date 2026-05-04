# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from gmdgen.gd.plans import ObjectPlan, SectionPlan
from gmdgen.gd.time_mapping import SpeedState
from gmdgen.generate.renderer import ObjectRenderer, render_plans_with_style


def _section(section_type: str = "drop") -> SectionPlan:
    return SectionPlan(0, 4, 0, 480, section_type, "cube", SpeedState.NORMAL, 0.8, 0.8, 0.8, 0.5)


def test_renderer_follows_reference_object_distribution() -> None:
    renderer = ObjectRenderer({"ids_by_class": {"decoration": ["777"]}}, safe_mode=True, seed=2)
    plan = ObjectPlan("1", 30, 200, "safe_decoration")

    rendered = renderer.render_object_plan(plan, _section("drop"))

    assert rendered.object_id in {"500", "501", "503", "504", "777"}


def test_renderer_varies_repeated_roles() -> None:
    plans = [ObjectPlan("999", idx * 30, 90, "beat_orb") for idx in range(6)]

    render_plans_with_style(plans, [_section("drop")], safe_mode=True, seed=4)

    assert len({plan.object_id for plan in plans}) > 1


def test_renderer_increases_drop_visual_accent() -> None:
    renderer = ObjectRenderer({}, safe_mode=True, seed=1)
    plan = ObjectPlan("500", 20, 200, "visual_accent_target", scale=1.0)

    renderer.render_object_plan(plan, _section("drop"))

    assert plan.scale >= 1.15


def test_renderer_respects_safe_mode() -> None:
    renderer = ObjectRenderer({"ids_by_class": {"decoration": ["999999"]}}, safe_mode=True, seed=1)
    plan = ObjectPlan("999999", 20, 200, "safe_decoration")

    renderer.render_object_plan(plan, _section("normal"))

    assert plan.object_id != "999999"


def test_renderer_does_not_generate_local_level_without_ollama_plan() -> None:
    renderer = ObjectRenderer({}, safe_mode=True, seed=1)

    assert renderer.render_decoration_cluster(0, 200, _section("break"), count=3)
