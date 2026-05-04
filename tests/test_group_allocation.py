from __future__ import annotations

from gmdgen.gd.plans import ObjectPlan, SectionPlan, TriggerPlan
from gmdgen.gd.time_mapping import SpeedState
from gmdgen.generate.group_allocation import allocate_trigger_target_groups


def _sections() -> list[SectionPlan]:
    return [
        SectionPlan(0, 2, 0, 480, "intro", "cube", SpeedState.NORMAL, 0.3, 0.2, 0.2, 0.2),
        SectionPlan(2, 4, 480, 960, "drop", "cube", SpeedState.NORMAL, 0.9, 0.8, 0.8, 0.7),
    ]


def test_group_allocator_creates_section_groups() -> None:
    objects = [ObjectPlan("500", 520, 240, "visual_accent_target")]
    triggers = [TriggerPlan("pulse", "1006", 520, 300, duration=0.1)]

    report = allocate_trigger_target_groups(objects, triggers, section_plans=_sections(), max_group_id=99)

    assert report.created_section_groups == 1
    assert objects[0].group_ids
    assert triggers[0].target_group == objects[0].group_ids[0]


def test_trigger_missing_target_group_auto_assigned() -> None:
    objects = [ObjectPlan("500", 520, 240, "visual_accent_target", group_ids=[8])]
    triggers = [TriggerPlan("pulse", "1006", 520, 300, duration=0.1)]

    report = allocate_trigger_target_groups(objects, triggers, section_plans=_sections(), max_group_id=99)

    assert report.auto_assigned_target_group_count == 1
    assert triggers[0].target_group == 8


def test_trigger_missing_target_group_removed_only_if_unresolvable() -> None:
    triggers = [TriggerPlan("pulse", "1006", 520, 300, duration=0.1)]

    report = allocate_trigger_target_groups([], triggers, section_plans=_sections(), max_group_id=99)

    assert report.unresolved_missing_target_group_count == 1
    assert triggers[0].target_group is None


def test_no_orphan_triggers_after_group_allocation() -> None:
    objects = [ObjectPlan("1", 100, 90, "ai_structure")]
    triggers = [TriggerPlan("move", "901", 100, 300, target_group=42, duration=0.2)]

    report = allocate_trigger_target_groups(objects, triggers, section_plans=_sections(), max_group_id=99)

    assert report.repaired_orphan_trigger_count == 1
    assert 42 in objects[0].group_ids
