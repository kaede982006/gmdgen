from __future__ import annotations

from gmdgen.gd.plans import TriggerPlan
from gmdgen.render.trigger_materializer import (
    TriggerIntent,
    materialize_trigger_intent,
    materialize_trigger_plan,
)


def test_trigger_intent_materialized_to_valid_trigger_plan() -> None:
    plan = materialize_trigger_intent(
        TriggerIntent("pulse", purpose="drop_accent", target_role="decoration_group", intensity=0.8, duration_hint=0.16),
        x=120,
        y=300,
        target_group=7,
        safe_mode=True,
    )

    assert plan is not None
    assert plan.trigger_type == "pulse"
    assert plan.object_id == "1006"
    assert plan.target_group == 7
    assert plan.duration > 0


def test_trigger_materializer_assigns_only_allowed_properties() -> None:
    plan = TriggerPlan("pulse", "0", 120, 300, target_group=1, properties={"move_x": 10, "color_channel": 2})

    materialized = materialize_trigger_plan(plan, safe_mode=True)

    assert materialized is not None
    assert materialized.properties == {"color_channel": 2}


def test_pulse_trigger_no_move_properties_after_materialization() -> None:
    plan = TriggerPlan("pulse", "1006", 120, 300, target_group=1, properties={"move_x": 5, "move_y": 5})

    materialize_trigger_plan(plan, safe_mode=True)

    assert "move_x" not in plan.properties
    assert "move_y" not in plan.properties


def test_color_trigger_no_move_properties_after_materialization() -> None:
    plan = TriggerPlan("color", "29", 120, 300, target_group=1, properties={"move_x": 5, "color_channel": 3})

    materialize_trigger_plan(plan, safe_mode=True)

    assert "move_x" not in plan.properties
    assert plan.properties["color_channel"] == 3


def test_move_trigger_no_color_properties_after_materialization() -> None:
    plan = TriggerPlan("move", "901", 120, 300, target_group=1, properties={"color_channel": 3, "move_x": 10})

    materialize_trigger_plan(plan, safe_mode=True)

    assert "color_channel" not in plan.properties
    assert plan.properties["move_x"] == 10
