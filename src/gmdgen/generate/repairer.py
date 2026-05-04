from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from gmdgen.features.tokenizer import (
    extract_object_field,
    extract_object_id,
    extract_object_number,
    parse_object_pairs,
    rewrite_object_xy,
)
from gmdgen.gd.plans import get_trigger_schema, is_trigger_allowed_in_mode

LOGGER = logging.getLogger(__name__)

# GD 속도 포탈 오브젝트 ID → (speed_label, min_x_gap)
# 속도 레이블은 이후 density_per_x_unit 계산에 사용
_SPEED_PORTAL_IDS: dict[str, tuple[str, int]] = {
    "200": ("half", 15),
    "201": ("normal", 20),
    "202": ("double", 28),
    "203": ("triple", 35),
    "1334": ("quadruple", 42),
}

# GD 게임모드 포탈 오브젝트 ID
_GAMEMODE_PORTAL_IDS: frozenset[str] = frozenset(
    {"12", "13", "47", "111", "660", "745", "1331"}
)

# GD 그룹 ID를 담는 오브젝트 키
_GROUP_KEY = "155"

# GD 트리거가 target group ID를 담는 키
_TRIGGER_TARGET_KEY = "51"

# trigger로 간주하는 오브젝트 ID 집합 (color/move/alpha/toggle 등)
_TRIGGER_IDS: frozenset[str] = frozenset(
    {
        "29",   # Color trigger
        "30",   # Move trigger
        "32",   # Alpha trigger
        "33",   # Toggle trigger
        "142",  # Spawn trigger
        "105",  # Pulse trigger
        "200",  # Shake trigger (color channel trigger)
        "899",  # Stop trigger
        "901",  # Move trigger in newer object maps
        "1006", "1007", "1049", "1268", "1346", "1347",
        "1520", "1595", "1611", "1616", "1815", "1817", "2067",
    }
)
_SAFE_TRIGGER_IDS: frozenset[str] = frozenset({"29", "32", "33", "899", "901", "1006", "1007", "1268"})
_TRIGGER_TYPE_BY_ID: dict[str, str] = {
    "901": "move",
    "1006": "pulse",
    "1007": "alpha",
    "1268": "spawn",
    "899": "stop",
    "1347": "follow",
    "1520": "shake",
    "29": "color",
    "33": "toggle",
    "1611": "count",
    "1815": "collision",
    "1817": "pickup",
}

# GD 격자 단위 (기본 오브젝트 배치 간격)
_GRID_UNIT = 30

# 단위 X 구간(grid 1칸)당 최대 오브젝트 수 (과밀 기준)
_DEFAULT_MAX_DENSITY_PER_GRID = 8


@dataclass
class RepairReport:
    x_monotone_fixed: int = 0
    group_id_remapped: int = 0
    orphan_trigger_removed: int = 0
    density_spread: int = 0
    grid_snapped: int = 0
    budget_pruned: int = 0
    unsafe_trigger_removed: int = 0
    group_bounds_fixed: int = 0
    trigger_schema_removed: int = 0
    trigger_schema_repaired: int = 0
    playability_pruned: int = 0
    k95_synced: bool = False
    issues_before: list[str] = field(default_factory=list)
    issues_after: list[str] = field(default_factory=list)

    @property
    def total_fixed(self) -> int:
        return (
            self.x_monotone_fixed
            + self.group_id_remapped
            + self.orphan_trigger_removed
            + self.density_spread
            + self.grid_snapped
            + self.budget_pruned
            + self.unsafe_trigger_removed
            + self.group_bounds_fixed
            + self.trigger_schema_removed
            + self.trigger_schema_repaired
            + self.playability_pruned
        )


def _parse_group_ids(level_object: str) -> list[int]:
    raw = extract_object_field(level_object, _GROUP_KEY)
    if not raw:
        return []
    group_ids: list[int] = []
    for part in raw.split("."):
        part = part.strip()
        if part.isdigit():
            group_ids.append(int(part))
    return group_ids


def _get_trigger_target(level_object: str) -> int | None:
    raw = extract_object_field(level_object, _TRIGGER_TARGET_KEY)
    if raw is None:
        return None
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    return None


def _is_trigger(object_id: str | None) -> bool:
    return object_id in _TRIGGER_IDS


def _set_field(level_object: str, key: str, value: str) -> str:
    pairs = parse_object_pairs(level_object)
    for idx, (pair_key, _) in enumerate(pairs):
        if pair_key == key:
            pairs[idx] = (key, value)
            flattened: list[str] = []
            for k, v in pairs:
                flattened.extend([k, v])
            return ",".join(flattened)
    pairs.append((key, value))
    flattened = []
    for k, v in pairs:
        flattened.extend([k, v])
    return ",".join(flattened)


def _remove_field(level_object: str, key: str) -> str:
    pairs = [pair for pair in parse_object_pairs(level_object) if pair[0] != key]
    flattened: list[str] = []
    for k, v in pairs:
        flattened.extend([k, v])
    return ",".join(flattened)


# ─────────────────────────────────────────────
# Step 1: X-monotone 강제
# Ch.4 §4.4 Constrained Optimization — x[i] >= x[i-1] + min_gap
# ─────────────────────────────────────────────

def repair_x_monotone(
    objects: list[str],
    *,
    min_gap: int = 1,
) -> tuple[list[str], int]:
    repaired: list[str] = []
    fixed_count = 0
    prev_x: int | None = None

    for obj in objects:
        x_val = extract_object_number(obj, "2")
        y_val = extract_object_number(obj, "3")

        if x_val is None or y_val is None:
            repaired.append(obj)
            continue

        x = int(round(x_val))
        y = int(round(y_val))

        if prev_x is not None and x < prev_x + min_gap:
            x = prev_x + min_gap
            obj = rewrite_object_xy(obj, x=x, y=y)
            fixed_count += 1

        prev_x = x
        repaired.append(obj)

    return repaired, fixed_count


# ─────────────────────────────────────────────
# Step 2: Group ID 전역 재할당
# 청크 간 충돌하는 group ID를 레벨 전역에서 고유하게 재부여
# Ch.16 §16.2 "Using Graphs to Describe Model Structure"
# ─────────────────────────────────────────────

def repair_group_ids(objects: list[str]) -> tuple[list[str], int]:
    existing_groups: set[int] = set()
    for obj in objects:
        for gid in _parse_group_ids(obj):
            existing_groups.add(gid)

    collision_sets: list[list[int]] = []
    seen_in_current: set[int] = set()
    last_group_set: set[int] = set()

    for obj in objects:
        obj_groups = set(_parse_group_ids(obj))
        if not obj_groups:
            continue
        overlap = obj_groups & last_group_set
        if overlap:
            collision_sets.append(sorted(obj_groups))
        seen_in_current.update(obj_groups)
        last_group_set = obj_groups

    if not collision_sets:
        return objects, 0

    # 충돌 그룹 ID를 새 고유 ID로 매핑
    remap: dict[int, int] = {}
    next_id = max(existing_groups, default=0) + 1
    for gid_set in collision_sets:
        for gid in gid_set:
            if gid not in remap:
                remap[gid] = next_id
                next_id += 1

    if not remap:
        return objects, 0

    repaired: list[str] = []
    fixed_count = 0
    for obj in objects:
        raw_group = extract_object_field(obj, _GROUP_KEY)
        if not raw_group:
            repaired.append(obj)
            continue

        parts = raw_group.split(".")
        new_parts: list[str] = []
        changed = False
        for part in parts:
            part = part.strip()
            if part.isdigit() and int(part) in remap:
                new_parts.append(str(remap[int(part)]))
                changed = True
            else:
                new_parts.append(part)

        if changed:
            obj = _set_field(obj, _GROUP_KEY, ".".join(new_parts))
            fixed_count += 1

        repaired.append(obj)

    return repaired, fixed_count


# ─────────────────────────────────────────────
# Step 3: Orphan trigger 삭제
# target group이 레벨 내에 존재하지 않는 trigger 제거
# Ch.16 "Structured Probabilistic Models" — 잘못된 edge 제거
# ─────────────────────────────────────────────

def repair_orphan_triggers(objects: list[str]) -> tuple[list[str], int]:
    # 레벨 내 모든 group ID 수집
    defined_groups: set[int] = set()
    for obj in objects:
        for gid in _parse_group_ids(obj):
            defined_groups.add(gid)

    repaired: list[str] = []
    removed_count = 0

    for obj in objects:
        obj_id = extract_object_id(obj)
        if not _is_trigger(obj_id):
            repaired.append(obj)
            continue

        target = _get_trigger_target(obj)
        if target is None:
            repaired.append(obj)
            continue

        if target not in defined_groups:
            removed_count += 1
            LOGGER.debug(
                "Remove orphan trigger id=%s target_group=%d (not in level)",
                obj_id,
                target,
            )
            continue

        repaired.append(obj)

    return repaired, removed_count


# ─────────────────────────────────────────────
# Step 4: 과밀 구간 완화
# 단위 grid 구간당 오브젝트 수가 max_per_grid 초과하면 분산
# Ch.7 §7.4 Dataset Augmentation — regularization as spreading
# ─────────────────────────────────────────────

def repair_density(
    objects: list[str],
    *,
    max_per_grid: int = _DEFAULT_MAX_DENSITY_PER_GRID,
    grid_unit: int = _GRID_UNIT,
) -> tuple[list[str], int]:
    bucket_counts: dict[int, int] = defaultdict(int)
    for obj in objects:
        x_val = extract_object_number(obj, "2")
        if x_val is not None:
            bucket = int(x_val) // grid_unit
            bucket_counts[bucket] += 1

    overcrowded = {b for b, c in bucket_counts.items() if c > max_per_grid}
    if not overcrowded:
        return objects, 0

    repaired: list[str] = []
    spread_count = 0
    bucket_seen: dict[int, int] = defaultdict(int)

    for obj in objects:
        x_val = extract_object_number(obj, "2")
        y_val = extract_object_number(obj, "3")
        if x_val is None or y_val is None:
            repaired.append(obj)
            continue

        x = int(round(x_val))
        bucket = x // grid_unit

        if bucket in overcrowded:
            slot = bucket_seen[bucket]
            if slot >= max_per_grid:
                extra_buckets = slot // max_per_grid
                new_x = (bucket + extra_buckets + 1) * grid_unit + (slot % max_per_grid)
                obj = rewrite_object_xy(obj, x=new_x, y=int(round(y_val)))
                spread_count += 1

        bucket_seen[bucket] += 1
        repaired.append(obj)

    return repaired, spread_count


# ─────────────────────────────────────────────
# Step 5: Grid snap
# x, y를 grid_unit 단위로 반올림 (GD 편집기 기본 격자 = 30)
# ─────────────────────────────────────────────

def repair_grid_snap(
    objects: list[str],
    *,
    grid_unit: int = _GRID_UNIT,
    snap_x: bool = True,
    snap_y: bool = False,
) -> tuple[list[str], int]:
    repaired: list[str] = []
    snapped_count = 0

    for obj in objects:
        x_val = extract_object_number(obj, "2")
        y_val = extract_object_number(obj, "3")
        if x_val is None or y_val is None:
            repaired.append(obj)
            continue

        x = int(round(x_val))
        y = int(round(y_val))
        changed = False

        if snap_x:
            snapped_x = round(x / grid_unit) * grid_unit
            if snapped_x != x:
                x = snapped_x
                changed = True

        if snap_y:
            snapped_y = round(y / grid_unit) * grid_unit
            if snapped_y != y:
                y = snapped_y
                changed = True

        if changed:
            obj = rewrite_object_xy(obj, x=x, y=y)
            snapped_count += 1

        repaired.append(obj)

    return repaired, snapped_count


# ─────────────────────────────────────────────
# Step 6: duplicate 제거
# 동일 (x, y, id) 오브젝트 중복 제거
# ─────────────────────────────────────────────

def repair_duplicates(objects: list[str]) -> tuple[list[str], int]:
    seen: set[tuple[str, int, int]] = set()
    repaired: list[str] = []
    removed_count = 0

    for obj in objects:
        obj_id = extract_object_id(obj) or ""
        x_val = extract_object_number(obj, "2")
        y_val = extract_object_number(obj, "3")
        if x_val is None or y_val is None:
            repaired.append(obj)
            continue

        key = (obj_id, int(round(x_val)), int(round(y_val)))
        if key in seen:
            removed_count += 1
            continue

        seen.add(key)
        repaired.append(obj)

    return repaired, removed_count


def prune_object_budget(objects: list[str], *, object_budget: int | None) -> tuple[list[str], int]:
    if object_budget is None or object_budget < 1 or len(objects) <= object_budget:
        return objects, 0

    def priority(obj: str) -> tuple[int, float]:
        obj_id = extract_object_id(obj)
        x_val = extract_object_number(obj, "2") or 0.0
        if obj_id in _SPEED_PORTAL_IDS:
            return (0, x_val)
        if _is_trigger(obj_id):
            return (1, x_val)
        if extract_object_field(obj, _GROUP_KEY):
            return (2, x_val)
        return (3, x_val)

    keep = sorted(objects, key=priority)[:object_budget]
    keep_sorted = sorted(keep, key=lambda obj: extract_object_number(obj, "2") or 0.0)
    return keep_sorted, len(objects) - len(keep_sorted)


def filter_unsafe_triggers(
    objects: list[str],
    *,
    safe_mode: bool,
    allowed_trigger_ids: frozenset[str] = _SAFE_TRIGGER_IDS,
) -> tuple[list[str], int]:
    if not safe_mode:
        return objects, 0
    repaired: list[str] = []
    removed = 0
    for obj in objects:
        obj_id = extract_object_id(obj)
        if obj_id in _SPEED_PORTAL_IDS:
            repaired.append(obj)
            continue
        if _is_trigger(obj_id) and obj_id not in allowed_trigger_ids:
            removed += 1
            continue
        repaired.append(obj)
    return repaired, removed


def repair_trigger_schema(
    objects: list[str],
    *,
    safe_mode: bool,
    max_group_id: int | None = None,
) -> tuple[list[str], int, int]:
    mode = "safe" if safe_mode else "advanced"
    repaired: list[str] = []
    removed = 0
    fixed = 0
    for obj in objects:
        obj_id = extract_object_id(obj)
        if obj_id in _SPEED_PORTAL_IDS or not _is_trigger(obj_id):
            repaired.append(obj)
            continue
        trigger_type = _TRIGGER_TYPE_BY_ID.get(str(obj_id))
        schema = get_trigger_schema(trigger_type or "")
        if schema is None or not is_trigger_allowed_in_mode(schema.trigger_type, mode):
            removed += 1
            continue
        target = _get_trigger_target(obj)
        if schema.requires_target_group and target is None:
            removed += 1
            continue
        if target is not None and max_group_id is not None and not (1 <= target <= max_group_id):
            removed += 1
            continue
        for field_name, (low, high) in schema.valid_ranges.items():
            key = "10" if field_name == "duration" else "63" if field_name == "spawn_delay" else ""
            if not key:
                continue
            value = extract_object_number(obj, key)
            if value is None:
                if field_name in schema.default_values:
                    obj = _set_field(obj, key, str(schema.default_values[field_name]))
                    fixed += 1
                continue
            clamped = max(low, min(high, float(value)))
            if abs(clamped - value) > 1e-9:
                obj = _set_field(obj, key, str(round(clamped, 5)))
                fixed += 1
        repaired.append(obj)
    return repaired, removed, fixed


def repair_group_bounds(
    objects: list[str],
    *,
    max_group_id: int | None,
) -> tuple[list[str], int]:
    if max_group_id is None or max_group_id < 1:
        return objects, 0
    remap: dict[int, int] = {}
    next_id = 1
    for obj in objects:
        for group_id in _parse_group_ids(obj):
            if group_id <= max_group_id:
                continue
            while next_id in remap.values() and next_id <= max_group_id:
                next_id += 1
            if next_id > max_group_id:
                remap[group_id] = max_group_id
            else:
                remap[group_id] = next_id
                next_id += 1
    if not remap:
        return objects, 0

    repaired: list[str] = []
    changed = 0
    for obj in objects:
        raw_group = extract_object_field(obj, _GROUP_KEY)
        if not raw_group:
            repaired.append(obj)
            continue
        parts = []
        did_change = False
        for part in raw_group.split("."):
            stripped = part.strip()
            if stripped.isdigit() and int(stripped) in remap:
                parts.append(str(remap[int(stripped)]))
                did_change = True
            else:
                parts.append(stripped)
        if did_change:
            obj = _set_field(obj, _GROUP_KEY, ".".join(parts))
            changed += 1
        repaired.append(obj)
    return repaired, changed


def prune_impossible_spacing(
    objects: list[str],
    *,
    min_gap: float,
) -> tuple[list[str], int]:
    if min_gap <= 0:
        return objects, 0
    kept: list[str] = []
    removed = 0
    last_gameplay_x: float | None = None
    for obj in sorted(objects, key=lambda raw: extract_object_number(raw, "2") or 0.0):
        obj_id = extract_object_id(obj)
        x_val = extract_object_number(obj, "2")
        if x_val is None or obj_id in _SPEED_PORTAL_IDS or _is_trigger(obj_id):
            kept.append(obj)
            continue
        if last_gameplay_x is not None and x_val - last_gameplay_x < min_gap:
            removed += 1
            continue
        last_gameplay_x = x_val
        kept.append(obj)
    return kept, removed


# ─────────────────────────────────────────────
# 전체 repair pipeline
# ─────────────────────────────────────────────

def repair_level_objects(
    objects: list[str],
    *,
    fix_x_monotone: bool = True,
    fix_group_ids: bool = True,
    fix_orphan_triggers: bool = True,
    fix_density: bool = True,
    fix_grid_snap: bool = True,
    fix_duplicates: bool = True,
    x_min_gap: int = 1,
    max_density_per_grid: int = _DEFAULT_MAX_DENSITY_PER_GRID,
    grid_unit: int = _GRID_UNIT,
    snap_x: bool = True,
    snap_y: bool = False,
    object_budget: int | None = None,
    max_group_id: int | None = None,
    safe_mode: bool = False,
    min_playability_gap: float | None = None,
) -> tuple[list[str], RepairReport]:
    report = RepairReport()
    current = list(objects)

    # 1. 중복 제거 (가장 먼저 — 이후 작업의 부하 감소)
    if fix_duplicates:
        current, count = repair_duplicates(current)
        if count:
            LOGGER.debug("repair_duplicates: removed %d", count)

    current, removed, fixed = repair_trigger_schema(
        current,
        safe_mode=safe_mode,
        max_group_id=max_group_id,
    )
    report.trigger_schema_removed = removed
    report.trigger_schema_repaired = fixed

    current, count = filter_unsafe_triggers(current, safe_mode=safe_mode)
    report.unsafe_trigger_removed = count
    if count:
        LOGGER.debug("filter_unsafe_triggers: removed %d", count)

    # 2. 과밀 분산 (density → 그 이후 x_monotone이 한 번에 수정)
    if fix_density:
        current, count = repair_density(
            current,
            max_per_grid=max_density_per_grid,
            grid_unit=grid_unit,
        )
        report.density_spread = count
        if count:
            LOGGER.debug("repair_density: spread %d", count)

    if min_playability_gap is not None:
        current, count = prune_impossible_spacing(current, min_gap=min_playability_gap)
        report.playability_pruned = count
        if count:
            LOGGER.debug("prune_impossible_spacing: pruned %d", count)

    # 3. X-monotone 강제 (density 분산 이후 적용해야 역행 재발 없음)
    if fix_x_monotone:
        current, count = repair_x_monotone(current, min_gap=x_min_gap)
        report.x_monotone_fixed = count
        if count:
            LOGGER.debug("repair_x_monotone: fixed %d", count)

    # 4. grid snap (x_monotone 이후 — snap이 역행을 다시 만들지 않도록)
    if fix_grid_snap:
        current, count = repair_grid_snap(
            current,
            grid_unit=grid_unit,
            snap_x=snap_x,
            snap_y=snap_y,
        )
        report.grid_snapped = count
        if count:
            LOGGER.debug("repair_grid_snap: snapped %d", count)

    # 5. Group ID 재할당 (위치 수정 완료 후 관계 수정)
    if fix_group_ids:
        current, count = repair_group_ids(current)
        report.group_id_remapped = count
        if count:
            LOGGER.debug("repair_group_ids: remapped %d", count)

    current, count = repair_group_bounds(current, max_group_id=max_group_id)
    report.group_bounds_fixed = count
    if count:
        LOGGER.debug("repair_group_bounds: fixed %d", count)

    # 6. Orphan trigger 제거 (group ID 재할당 이후)
    if fix_orphan_triggers:
        current, count = repair_orphan_triggers(current)
        report.orphan_trigger_removed = count
        if count:
            LOGGER.debug("repair_orphan_triggers: removed %d", count)

    current, count = prune_object_budget(current, object_budget=object_budget)
    report.budget_pruned = count
    if count:
        LOGGER.debug("prune_object_budget: pruned %d", count)

    report.k95_synced = True

    LOGGER.info(
        "repair_level_objects: total_fixed=%d "
        "(x_mono=%d group=%d trigger=%d density=%d snap=%d)",
        report.total_fixed,
        report.x_monotone_fixed,
        report.group_id_remapped,
        report.orphan_trigger_removed,
        report.density_spread,
        report.grid_snapped,
    )

    return current, report


def repair_report_to_dict(report: RepairReport) -> dict[str, Any]:
    return {
        "x_monotone_fixed": report.x_monotone_fixed,
        "group_id_remapped": report.group_id_remapped,
        "orphan_trigger_removed": report.orphan_trigger_removed,
        "density_spread": report.density_spread,
        "grid_snapped": report.grid_snapped,
        "budget_pruned": report.budget_pruned,
        "unsafe_trigger_removed": report.unsafe_trigger_removed,
        "group_bounds_fixed": report.group_bounds_fixed,
        "trigger_schema_removed": report.trigger_schema_removed,
        "trigger_schema_repaired": report.trigger_schema_repaired,
        "playability_pruned": report.playability_pruned,
        "total_fixed": report.total_fixed,
    }
