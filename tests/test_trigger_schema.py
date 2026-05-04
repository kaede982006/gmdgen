from __future__ import annotations

from gmdgen.gd.plans import (
    TriggerPlan,
    apply_trigger_defaults,
    get_trigger_schema,
    is_trigger_allowed_in_mode,
    validate_trigger_plan_schema,
)
from gmdgen.gd.triggers import (
    encode_trigger_properties_safe,
    list_supported_triggers,
    validate_trigger_properties,
)
from gmdgen.generate.repairer import repair_trigger_schema


def _trigger(trigger_type: str, object_id: str, *, target: int | None = 1, duration: float = 0.2) -> TriggerPlan:
    return TriggerPlan(
        trigger_type=trigger_type,
        object_id=object_id,
        x=120.0,
        y=240.0,
        target_group=target,
        duration=duration,
    )


def test_move_trigger_schema_valid() -> None:
    plan = _trigger("move", "901", duration=0.4)
    assert validate_trigger_plan_schema(plan, "advanced") == []


def test_pulse_trigger_schema_valid() -> None:
    plan = _trigger("pulse", "1006", duration=0.18)
    assert validate_trigger_plan_schema(plan, "safe") == []


def test_alpha_trigger_schema_valid() -> None:
    plan = _trigger("alpha", "1007", duration=0.18)
    assert validate_trigger_plan_schema(plan, "safe") == []


def test_spawn_trigger_requires_valid_delay() -> None:
    plan = TriggerPlan(
        trigger_type="spawn",
        object_id="1268",
        x=0.0,
        y=0.0,
        target_group=1,
        spawn_delay=99.0,
    )
    defaulted = apply_trigger_defaults(plan, "advanced")
    assert defaulted is not None
    assert defaulted.spawn_delay == 8.0


def test_stop_trigger_requires_target() -> None:
    plan = _trigger("stop", "899", target=None, duration=0.0)
    issues = validate_trigger_plan_schema(plan, "advanced")
    assert any("target_group" in issue for issue in issues)


def test_safe_mode_filters_advanced_trigger() -> None:
    assert is_trigger_allowed_in_mode("move", "safe") is True
    assert is_trigger_allowed_in_mode("follow", "safe") is False
    assert is_trigger_allowed_in_mode("pulse", "safe") is True


def test_unknown_trigger_is_rejected() -> None:
    assert get_trigger_schema("does_not_exist") is None
    plan = _trigger("does_not_exist", "9999")
    assert validate_trigger_plan_schema(plan, "advanced")


def test_trigger_defaults_are_applied() -> None:
    plan = _trigger("pulse", "1006", duration=0.0)
    defaulted = apply_trigger_defaults(plan, "safe")
    assert defaulted is not None
    assert defaulted.duration > 0.0


def test_invalid_duration_is_repaired_or_rejected() -> None:
    objects = [
        "1,1,2,10,3,10,155,1",
        "1,1006,2,20,3,20,51,1,10,99",
        "1,1347,2,30,3,20,51,1",
    ]
    repaired, removed, fixed = repair_trigger_schema(objects, safe_mode=True, max_group_id=10)
    assert removed == 1
    assert fixed == 1
    assert any("1,1006" in obj and "10,2" in obj for obj in repaired)


def test_trigger_schema_registry_lists_safe_triggers() -> None:
    safe = list_supported_triggers("safe")
    for trigger_type in ["move", "pulse", "alpha", "spawn", "stop", "color", "toggle"]:
        assert trigger_type in safe
    assert "follow" not in safe


def test_trigger_schema_registry_lists_advanced_triggers() -> None:
    advanced = list_supported_triggers("advanced")
    assert "follow" in advanced
    assert "collision" in advanced


def test_unsupported_trigger_rejected_in_safe_mode() -> None:
    plan = _trigger("follow", "1347", target=1, duration=0.2)
    assert any("safe" in issue for issue in validate_trigger_properties(plan, "safe"))


def test_unknown_trigger_property_rejected() -> None:
    plan = _trigger("pulse", "1006", target=1, duration=0.2)
    plan.properties["mystery"] = 1
    assert any("unknown_trigger_property" in issue for issue in validate_trigger_properties(plan, "safe"))


def test_trigger_property_defaults_applied() -> None:
    plan = _trigger("spawn", "1268", target=1, duration=0.0)
    encoded = encode_trigger_properties_safe(plan, "safe")
    assert encoded["1"] == "1268"
    assert encoded["63"] == 0.0


def test_move_trigger_property_ranges() -> None:
    plan = _trigger("move", "901", target=1, duration=99.0)
    encoded = encode_trigger_properties_safe(plan, "safe")
    assert encoded["10"] == 8.0


def test_pulse_trigger_color_properties() -> None:
    plan = _trigger("pulse", "1006", target=1, duration=0.2)
    encoded = encode_trigger_properties_safe(plan, "safe")
    assert encoded["1"] == "1006"
    assert "51" in encoded


def test_alpha_trigger_opacity_range() -> None:
    plan = _trigger("alpha", "1007", target=1, duration=0.2)
    plan.properties["opacity"] = 2.0
    assert any("above_max" in issue for issue in validate_trigger_properties(plan, "safe"))


def test_spawn_trigger_delay_range() -> None:
    plan = TriggerPlan("spawn", "1268", x=0, y=0, target_group=1, spawn_delay=99.0)
    encoded = encode_trigger_properties_safe(plan, "safe")
    assert encoded["63"] == 8.0


def test_color_trigger_channel_bounds() -> None:
    plan = _trigger("color", "29", target=1, duration=0.1)
    encoded = encode_trigger_properties_safe(plan, "safe")
    assert encoded["1"] == "29"


def test_safe_encoder_does_not_emit_unknown_keys() -> None:
    plan = _trigger("move", "901", target=1, duration=0.2)
    encoded = encode_trigger_properties_safe(plan, "safe")
    assert None not in encoded
    assert all(key in {"1", "2", "3", "10", "35", "51", "58"} for key in encoded)
