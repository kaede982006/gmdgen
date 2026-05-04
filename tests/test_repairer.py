from __future__ import annotations

from gmdgen.generate.repairer import (
    filter_unsafe_triggers,
    prune_object_budget,
    repair_density,
    repair_duplicates,
    repair_grid_snap,
    repair_group_ids,
    repair_level_objects,
    repair_orphan_triggers,
    repair_x_monotone,
)


def _obj(object_id: str, x: int, y: int, extra: str = "") -> str:
    base = f"1,{object_id},2,{x},3,{y}"
    return f"{base},{extra}" if extra else base


def _obj_with_group(object_id: str, x: int, y: int, group_id: int) -> str:
    return f"1,{object_id},2,{x},3,{y},155,{group_id}"


def _trigger(target_group: int, x: int = 10, y: int = 10) -> str:
    return f"1,29,2,{x},3,{y},51,{target_group}"


# ── X-monotone ────────────────────────────────────────────────────────────────

def test_repair_x_monotone_fixes_reversed() -> None:
    objects = [_obj("1", 100, 10), _obj("1", 50, 10), _obj("1", 200, 10)]
    repaired, count = repair_x_monotone(objects, min_gap=1)

    xs = [
        float(part.split(",")[3])
        for part in repaired
        for key, val in zip(part.split(",")[::2], part.split(",")[1::2])
        if key == "2"
    ]
    assert count == 1

    from gmdgen.features.tokenizer import extract_object_number
    x_values = [extract_object_number(obj, "2") for obj in repaired]
    assert all(x is not None for x in x_values)
    for prev_x, next_x in zip(x_values, x_values[1:]):
        assert next_x >= prev_x  # type: ignore[operator]


def test_repair_x_monotone_no_change_when_sorted() -> None:
    objects = [_obj("1", 10, 5), _obj("1", 20, 5), _obj("1", 30, 5)]
    _, count = repair_x_monotone(objects)
    assert count == 0


# ── Orphan triggers ───────────────────────────────────────────────────────────

def test_repair_orphan_triggers_removes_dangling() -> None:
    objects = [
        _obj_with_group("1", 10, 10, group_id=5),
        _trigger(target_group=5),   # valid
        _trigger(target_group=99),  # orphan — group 99 not defined
    ]
    repaired, count = repair_orphan_triggers(objects)
    assert count == 1
    assert len(repaired) == 2


def test_repair_orphan_triggers_keeps_valid() -> None:
    objects = [
        _obj_with_group("1", 10, 10, group_id=3),
        _trigger(target_group=3),
    ]
    repaired, count = repair_orphan_triggers(objects)
    assert count == 0
    assert len(repaired) == 2


# ── Duplicates ────────────────────────────────────────────────────────────────

def test_repair_duplicates_removes_identical() -> None:
    objects = [
        _obj("1", 30, 30),
        _obj("1", 30, 30),  # duplicate
        _obj("2", 60, 30),
    ]
    repaired, count = repair_duplicates(objects)
    assert count == 1
    assert len(repaired) == 2


def test_repair_duplicates_keeps_different_id() -> None:
    objects = [_obj("1", 30, 30), _obj("2", 30, 30)]
    _, count = repair_duplicates(objects)
    assert count == 0


# ── Grid snap ─────────────────────────────────────────────────────────────────

def test_repair_grid_snap_x_aligned() -> None:
    objects = [_obj("1", 25, 10), _obj("1", 55, 10)]
    repaired, count = repair_grid_snap(objects, grid_unit=30, snap_x=True, snap_y=False)
    from gmdgen.features.tokenizer import extract_object_number
    assert count == 2
    x_values = [extract_object_number(obj, "2") for obj in repaired]
    assert x_values[0] == 30
    assert x_values[1] == 60


# ── Density ───────────────────────────────────────────────────────────────────

def test_repair_density_spreads_overcrowded() -> None:
    objects = [_obj("1", 30, 10)] * 12  # 12 objects at same grid bucket
    repaired, count = repair_density(objects, max_per_grid=8, grid_unit=30)
    assert count > 0
    assert len(repaired) == 12


# ── Full pipeline ─────────────────────────────────────────────────────────────

def test_full_repair_pipeline() -> None:
    objects = [
        _obj_with_group("1", 90, 10, group_id=1),
        _obj("1", 30, 10),           # x 역행
        _obj("1", 60, 10),
        _trigger(target_group=99),   # orphan trigger
        _obj("2", 120, 10),
        _obj("2", 120, 10),          # duplicate
    ]
    repaired, report = repair_level_objects(
        objects,
        fix_x_monotone=True,
        fix_group_ids=True,
        fix_orphan_triggers=True,
        fix_density=True,
        fix_grid_snap=False,
        fix_duplicates=True,
        x_min_gap=1,
    )
    assert report.x_monotone_fixed >= 1
    assert report.orphan_trigger_removed == 1
    assert report.total_fixed >= 2
    assert len(repaired) < len(objects)


def test_repair_pipeline_empty_input() -> None:
    repaired, report = repair_level_objects([])
    assert repaired == []
    assert report.total_fixed == 0


def test_repairer_respects_object_budget() -> None:
    objects = [_obj("1", x * 30, 10) for x in range(10)]
    repaired, pruned = prune_object_budget(objects, object_budget=4)
    assert len(repaired) == 4
    assert pruned == 6


def test_safe_mode_filters_unsafe_triggers() -> None:
    objects = [
        _obj("1", 10, 10),
        _obj("200", 15, 20),        # speed portal must not be removed
        "1,1347,2,20,3,20,51,1",  # advanced follow trigger in safe mode
        "1,1006,2,30,3,20,51,1",  # safe pulse trigger
    ]
    repaired, removed = filter_unsafe_triggers(objects, safe_mode=True)
    assert removed == 1
    assert all("1,1347" not in obj for obj in repaired)
    assert any(obj.startswith("1,200,") for obj in repaired)
