from __future__ import annotations

from gmdgen.gd.plans import ObjectPlan, SectionPlan
from gmdgen.gd.time_mapping import SpeedState
from gmdgen.generate.playability import repair_playability_plans


def _drop_section() -> SectionPlan:
    return SectionPlan(0, 4, 0, 720, "drop", "cube", SpeedState.NORMAL, 0.9, 0.8, 0.8, 0.7)


def test_playability_repair_improves_score() -> None:
    section = _drop_section()
    objects = [ObjectPlan("36", 100 + idx * 10, 180, "beat_orb") for idx in range(8)]

    report = repair_playability_plans(objects, [section], difficulty="easy")

    assert report.score_after >= report.score_before
    assert report.simplified_dense_orb_chain > 0 or report.converted_gameplay_to_decoration > 0


def test_playability_repair_does_not_empty_drop_section() -> None:
    section = _drop_section()
    objects = [ObjectPlan("36", 100 + idx * 10, 180, "beat_orb") for idx in range(5)]

    repair_playability_plans(objects, [section], difficulty="easy")

    assert any(section.start_x <= obj.x <= section.end_x for obj in objects)
    assert any(obj.role in {"visual_accent_target", "safe_decoration", "beat_orb"} for obj in objects)


def test_dense_orb_chain_simplified() -> None:
    section = _drop_section()
    objects = [ObjectPlan("36", 100 + idx * 12, 180, "beat_orb") for idx in range(6)]

    report = repair_playability_plans(objects, [section], difficulty="normal")

    assert report.simplified_dense_orb_chain >= 1


def test_unsafe_gameplay_converted_to_decoration() -> None:
    section = _drop_section()
    objects = [
        ObjectPlan("8", 100, 90, "obstacle"),
        ObjectPlan("8", 110, 90, "obstacle"),
        ObjectPlan("8", 120, 90, "obstacle"),
    ]

    report = repair_playability_plans(objects, [section], difficulty="easy")

    assert report.converted_gameplay_to_decoration >= 1
    assert any(obj.role == "visual_accent_target" for obj in objects)


def test_portal_recovery_margin_added() -> None:
    section = _drop_section()
    objects = [
        ObjectPlan("202", 100, 150, "speed_portal"),
        ObjectPlan("36", 120, 180, "beat_orb"),
        ObjectPlan("8", 130, 90, "hazard"),
    ]

    report = repair_playability_plans(objects, [section], difficulty="normal")

    assert report.recovery_margin_adjusted + report.removed_hazards_after_portal >= 1
