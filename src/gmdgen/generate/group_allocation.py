from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gmdgen.gd.plans import ObjectPlan, SectionPlan, TriggerPlan, get_trigger_schema


@dataclass(slots=True)
class GroupAllocationReport:
    created_section_groups: int = 0
    auto_assigned_target_group_count: int = 0
    repaired_orphan_trigger_count: int = 0
    unresolved_missing_target_group_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_section_groups": self.created_section_groups,
            "auto_assigned_target_group_count": self.auto_assigned_target_group_count,
            "repaired_orphan_trigger_count": self.repaired_orphan_trigger_count,
            "unresolved_missing_target_group_count": self.unresolved_missing_target_group_count,
            "warnings": list(self.warnings),
        }


def allocate_trigger_target_groups(
    object_plans: list[ObjectPlan],
    trigger_plans: list[TriggerPlan],
    *,
    section_plans: list[SectionPlan] | None = None,
    max_group_id: int = 9999,
) -> GroupAllocationReport:
    report = GroupAllocationReport()
    allocator = _PlanGroupAllocator(object_plans, section_plans or [], max_group_id=max_group_id, report=report)
    for idx, trigger in enumerate(trigger_plans):
        schema = get_trigger_schema(trigger.trigger_type)
        if schema is None:
            continue
        requires_target = schema.requires_target_group or "target_group" in schema.required_fields
        if trigger.target_group is not None:
            if allocator.ensure_group_exists(trigger.target_group, trigger):
                report.repaired_orphan_trigger_count += 1
                report.warnings.append(f"repaired_orphan_trigger_target[{idx}]: {trigger.target_group}")
            continue
        if not requires_target and not _has_target_hint(trigger):
            continue
        group_id = allocator.group_for_trigger(trigger)
        if group_id is None:
            if requires_target:
                report.unresolved_missing_target_group_count += 1
                report.warnings.append(f"unresolved_missing_target_group[{idx}]: {trigger.trigger_type}")
            continue
        trigger.target_group = group_id
        report.auto_assigned_target_group_count += 1
        report.warnings.append(f"auto_assigned_target_group[{idx}]: {trigger.trigger_type}->{group_id}")
    return report


class _PlanGroupAllocator:
    def __init__(
        self,
        object_plans: list[ObjectPlan],
        section_plans: list[SectionPlan],
        *,
        max_group_id: int,
        report: GroupAllocationReport,
    ) -> None:
        self.object_plans = object_plans
        self.section_plans = section_plans
        self.max_group_id = max(1, int(max_group_id or 9999))
        self.report = report
        existing = [
            group_id
            for obj in object_plans
            for group_id in obj.group_ids
            if isinstance(group_id, int) and group_id > 0
        ]
        self._next_id = max(existing, default=0) + 1

    def group_for_trigger(self, trigger: TriggerPlan) -> int | None:
        section_id = self._section_id_for_trigger(trigger)
        objects = self._objects_for(section_id, trigger)
        if not objects:
            return None
        existing = self._first_group(objects)
        if existing is not None:
            return existing
        group_id = self._allocate()
        if group_id is None:
            return None
        for obj in objects:
            if group_id not in obj.group_ids:
                obj.group_ids.append(group_id)
        self.report.created_section_groups += 1
        return group_id

    def ensure_group_exists(self, group_id: int, trigger: TriggerPlan) -> bool:
        if not (1 <= int(group_id) <= self.max_group_id):
            return False
        if any(group_id in obj.group_ids for obj in self.object_plans):
            return False
        section_id = self._section_id_for_trigger(trigger)
        objects = self._objects_for(section_id, trigger) or self.object_plans[:1]
        if not objects:
            return False
        objects[0].group_ids.append(group_id)
        return True

    def _allocate(self) -> int | None:
        while self._next_id <= self.max_group_id:
            group_id = self._next_id
            self._next_id += 1
            if not any(group_id in obj.group_ids for obj in self.object_plans):
                return group_id
        return None

    def _objects_for(self, section_id: int | None, trigger: TriggerPlan) -> list[ObjectPlan]:
        candidates = [
            obj
            for obj in self.object_plans
            if section_id is None or self._section_id_for_object(obj) == section_id
        ]
        if not candidates:
            candidates = list(self.object_plans)
        target_role = str(trigger.properties.get("target_role", "") if isinstance(trigger.properties, dict) else "").lower()
        purpose = str(trigger.properties.get("purpose", "") if isinstance(trigger.properties, dict) else "").lower()
        if target_role in {"decoration_group", "background_group"} or purpose in {"drop_accent", "decoration", "visibility"}:
            filtered = [obj for obj in candidates if _is_decoration_role(obj.role)]
            return filtered or candidates[:1]
        if target_role == "gameplay_group":
            filtered = [obj for obj in candidates if _is_gameplay_role(obj.role)]
            return filtered or candidates[:1]
        return candidates[: max(1, min(4, len(candidates)))]

    def _first_group(self, objects: list[ObjectPlan]) -> int | None:
        for obj in objects:
            for group_id in obj.group_ids:
                if isinstance(group_id, int) and 1 <= group_id <= self.max_group_id:
                    return group_id
        return None

    def _section_id_for_trigger(self, trigger: TriggerPlan) -> int | None:
        if isinstance(trigger.properties, dict):
            value = _optional_int(trigger.properties.get("section_id"))
            if value is not None:
                return value
        return self._section_id_for_x(trigger.x)

    def _section_id_for_object(self, obj: ObjectPlan) -> int | None:
        section_id = obj.safety_flags.get("section_id") if isinstance(obj.safety_flags, dict) else None
        if isinstance(section_id, int):
            return section_id
        return self._section_id_for_x(obj.x)

    def _section_id_for_x(self, x_value: float | None) -> int | None:
        if x_value is None or not self.section_plans:
            return None
        best_idx = 0
        best_dist = float("inf")
        for idx, section in enumerate(self.section_plans):
            if section.start_x <= x_value <= section.end_x:
                return idx
            dist = min(abs(x_value - section.start_x), abs(x_value - section.end_x))
            if dist < best_dist:
                best_idx, best_dist = idx, dist
        return best_idx


def _has_target_hint(trigger: TriggerPlan) -> bool:
    if not isinstance(trigger.properties, dict):
        return False
    return any(trigger.properties.get(key) for key in ("target_role", "purpose", "section_id"))


def _is_decoration_role(role: Any) -> bool:
    text = str(role or "").lower()
    return "decor" in text or "accent" in text or "visual" in text or "background" in text


def _is_gameplay_role(role: Any) -> bool:
    text = str(role or "").lower()
    return "orb" in text or "pad" in text or "gameplay" in text or "structure" in text


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except Exception:
        return None
