# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from dataclasses import dataclass, field

from gmdgen.features.tokenizer import extract_object_field, extract_object_id, extract_object_number
from gmdgen.gd.plans import ObjectPlan, TriggerMode, TriggerPlan, plans_to_level_objects
from gmdgen.gd.triggers import validate_trigger_properties
from gmdgen.generate.validator import parse_save_string_safe, round_trip_validate, validate_encoder_safe_keys


@dataclass(slots=True)
class EditorSafetyReport:
    parse_ok: bool = False
    encode_ok: bool = False
    round_trip_ok: bool = False
    object_count_before: int = 0
    object_count_after: int = 0
    unsupported_trigger_count: int = 0
    unknown_key_count: int = 0
    nan_coordinate_count: int = 0
    invalid_group_count: int = 0
    invalid_property_count: int = 0
    warnings: list[str] = field(default_factory=list)
    fatal_errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.fatal_errors

    def to_dict(self) -> dict:
        return {
            "parse_ok": self.parse_ok,
            "encode_ok": self.encode_ok,
            "round_trip_ok": self.round_trip_ok,
            "object_count_before": self.object_count_before,
            "object_count_after": self.object_count_after,
            "unsupported_trigger_count": self.unsupported_trigger_count,
            "unknown_key_count": self.unknown_key_count,
            "nan_coordinate_count": self.nan_coordinate_count,
            "invalid_group_count": self.invalid_group_count,
            "invalid_property_count": self.invalid_property_count,
            "warnings": list(self.warnings),
            "fatal_errors": list(self.fatal_errors),
            "valid": self.valid,
            "score": editor_safety_score(self),
        }


_CONSERVATIVE_KEYS = {
    "1", "2", "3", "6", "10", "20", "24", "25", "32", "35", "51", "58", "63", "71", "155"
}


def validate_editor_safety(
    object_plans: list[ObjectPlan],
    trigger_plans: list[TriggerPlan],
    mode: TriggerMode | str | bool | None = TriggerMode.SAFE,
    *,
    max_group_id: int | None = None,
) -> EditorSafetyReport:
    report = EditorSafetyReport(parse_ok=True, encode_ok=True)
    for plan in object_plans:
        if not _finite(plan.x) or not _finite(plan.y):
            report.nan_coordinate_count += 1
            report.fatal_errors.append("nan_coordinate_in_object_plan")
        for group_id in plan.group_ids:
            if group_id <= 0 or (max_group_id is not None and group_id > max_group_id):
                report.invalid_group_count += 1
    for trigger in trigger_plans:
        issues = validate_trigger_properties(trigger, mode, max_group_id=max_group_id)
        for issue in issues:
            if "unsupported" in issue or "not_allowed" in issue:
                report.unsupported_trigger_count += 1
            else:
                report.invalid_property_count += 1
            report.warnings.append(issue)
    if report.nan_coordinate_count or report.invalid_group_count:
        report.fatal_errors.append("editor_safety_plan_validation_failed")
    try:
        objects = plans_to_level_objects(object_plans, trigger_plans, trigger_mode=mode)
        report.object_count_before = len(objects)
    except Exception as exc:  # noqa: BLE001
        report.encode_ok = False
        report.fatal_errors.append(f"encode_failed: {exc}")
    return report


def validate_save_string_safety(
    level_string: str,
    mode: TriggerMode | str | bool | None = TriggerMode.SAFE,
    *,
    max_group_id: int | None = None,
) -> EditorSafetyReport:
    safe_mode = mode == TriggerMode.SAFE or str(mode).lower() == "safe"
    objects, parse_issues = parse_save_string_safe(level_string)
    report = EditorSafetyReport(
        parse_ok=not parse_issues,
        object_count_before=len(objects),
    )
    if parse_issues:
        report.fatal_errors.extend(parse_issues)
        return report

    key_issues = validate_encoder_safe_keys(objects, safe_mode=safe_mode, max_group_id=max_group_id)
    for issue in key_issues:
        if "invalid_numeric_value" in issue:
            report.nan_coordinate_count += 1
            report.fatal_errors.append(issue)
        elif "unsupported_trigger" in issue:
            report.unsupported_trigger_count += 1
            report.fatal_errors.append(issue)
        elif "group_id_bounds" in issue:
            report.invalid_group_count += 1
            report.fatal_errors.append(issue)
        else:
            report.warnings.append(issue)

    for obj in objects:
        _inspect_raw_object(obj, report)
    round_trip = round_trip_validate(level_string, safe_mode=safe_mode, max_group_id=max_group_id)
    report.round_trip_ok = bool(round_trip["valid"])
    report.encode_ok = bool(round_trip["encode_ok"])
    report.object_count_after = int(round_trip["object_count_after"])
    for issue in round_trip["issues"]:
        text = str(issue)
        if "round_trip_count_mismatch" in text or "parse" in text or "unsupported" in text:
            report.fatal_errors.append(text)
        else:
            report.warnings.append(text)
    return report


def run_encoder_round_trip_safety_check(
    object_plans: list[ObjectPlan],
    trigger_plans: list[TriggerPlan],
    mode: TriggerMode | str | bool | None = TriggerMode.SAFE,
    *,
    max_group_id: int | None = None,
) -> EditorSafetyReport:
    plan_report = validate_editor_safety(
        object_plans,
        trigger_plans,
        mode,
        max_group_id=max_group_id,
    )
    if not plan_report.encode_ok or plan_report.fatal_errors:
        return plan_report
    objects = plans_to_level_objects(object_plans, trigger_plans, trigger_mode=mode)
    save_report = validate_save_string_safety(
        "kA11,0;" + ";".join(objects) + ";",
        mode,
        max_group_id=max_group_id,
    )
    save_report.warnings = plan_report.warnings + save_report.warnings
    save_report.fatal_errors = plan_report.fatal_errors + save_report.fatal_errors
    save_report.invalid_property_count += plan_report.invalid_property_count
    save_report.unsupported_trigger_count += plan_report.unsupported_trigger_count
    save_report.invalid_group_count += plan_report.invalid_group_count
    save_report.nan_coordinate_count += plan_report.nan_coordinate_count
    return save_report


def editor_safety_score(report: EditorSafetyReport) -> float:
    penalty = 0.0
    penalty += len(report.fatal_errors) * 0.35
    penalty += report.unsupported_trigger_count * 0.2
    penalty += report.unknown_key_count * 0.08
    penalty += report.nan_coordinate_count * 0.3
    penalty += report.invalid_group_count * 0.18
    penalty += report.invalid_property_count * 0.1
    penalty += len(report.warnings) * 0.03
    if not report.round_trip_ok:
        penalty += 0.2
    return max(0.0, 1.0 - min(1.0, penalty))


def editor_safety_report_to_dict(report: EditorSafetyReport) -> dict:
    return report.to_dict()


def _inspect_raw_object(obj: str, report: EditorSafetyReport) -> None:
    parts = obj.split(",")
    if len(parts) % 2 != 0:
        report.unknown_key_count += 1
        report.warnings.append("odd_object_field_count")
    for key, value in zip(parts[::2], parts[1::2]):
        if key not in _CONSERVATIVE_KEYS:
            report.unknown_key_count += 1
            report.warnings.append(f"unknown_or_unverified_key: {key}")
        if value.lower() in {"nan", "inf", "-inf"}:
            report.nan_coordinate_count += 1
            report.fatal_errors.append(f"invalid_numeric_value: {key}={value}")
    obj_id = extract_object_id(obj)
    if obj_id is None or not str(obj_id).isdigit() or int(obj_id) <= 0:
        report.fatal_errors.append("invalid_object_id")
    for group_key in ("51", "71", "155"):
        raw = extract_object_field(obj, group_key)
        if not raw:
            continue
        for part in raw.split("."):
            if not part.strip().isdigit() or int(part.strip()) <= 0:
                report.invalid_group_count += 1
                report.fatal_errors.append(f"invalid_group_id: {group_key}={raw}")
    for coord_key in ("2", "3"):
        value = extract_object_number(obj, coord_key)  # type: ignore
        if value is None:
            continue
        if not _finite(value):  # type: ignore
            report.nan_coordinate_count += 1
            report.fatal_errors.append(f"non_finite_coordinate: {coord_key}")


def _finite(value: float | int) -> bool:
    return math.isfinite(float(value))
