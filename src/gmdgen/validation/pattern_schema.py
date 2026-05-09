# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gmdgen.patterns.builder import PATTERNS_DIR
from gmdgen.types import VALID_DIFFICULTIES, VALID_GAME_MODES


PATTERN_INDEX_FILE_NAME = "patterns_index.json"
ALLOWED_TRIGGER_TYPES = {"move", "pulse", "alpha", "spawn", "stop", "toggle", "color"}


@dataclass(slots=True)
class PatternValidationIssue:
    path: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "code": self.code, "message": self.message}


@dataclass(slots=True)
class PatternLibraryValidationReport:
    checked_files: int = 0
    valid_patterns: int = 0
    index_files: int = 0
    invalid_patterns: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    repair_suggestions: list[str] = field(default_factory=list)
    destructive_changes: bool = False

    @property
    def passed(self) -> bool:
        return not self.invalid_patterns

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_files": self.checked_files,
            "valid_patterns": self.valid_patterns,
            "index_files": self.index_files,
            "invalid_patterns": list(self.invalid_patterns),
            "warnings": list(self.warnings),
            "repair_suggestions": list(self.repair_suggestions),
            "destructive_changes": self.destructive_changes,
            "passed": self.passed,
        }


def is_pattern_index_file(path: Path) -> bool:
    return path.name == PATTERN_INDEX_FILE_NAME


def validate_pattern_library(root: Path | None = None) -> PatternLibraryValidationReport:
    directory = root or PATTERNS_DIR
    report = PatternLibraryValidationReport()
    for path in sorted(directory.rglob("*.json")):
        report.checked_files += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report.invalid_patterns.append(
                {
                    "path": str(path),
                    "errors": [f"json_parse_failed:{exc.msg}"],
                }
            )
            report.repair_suggestions.append(f"Rebuild or fix malformed JSON: {path}")
            continue
        if is_pattern_index_file(path):
            report.index_files += 1
            errors, warnings = validate_patterns_index_payload(payload, path=path)
        else:
            errors, warnings = validate_pattern_object_payload(payload, path=path)
        report.warnings.extend(issue.to_dict() for issue in warnings)
        if errors:
            report.invalid_patterns.append(
                {
                    "path": str(path),
                    "errors": [issue.code for issue in errors],
                    "messages": [issue.message for issue in errors],
                }
            )
            report.repair_suggestions.append(f"Repair pattern schema violations in {path}")
        elif not is_pattern_index_file(path):
            report.valid_patterns += 1
    return report


def validate_patterns_index_payload(
    payload: Any,
    *,
    path: Path | str = PATTERN_INDEX_FILE_NAME,
) -> tuple[list[PatternValidationIssue], list[PatternValidationIssue]]:
    path_text = str(path)
    errors: list[PatternValidationIssue] = []
    warnings: list[PatternValidationIssue] = []
    if not isinstance(payload, dict):
        return [_issue(path_text, "index_must_be_object", "patterns_index.json must be a JSON object")], warnings
    cells = payload.get("cells")
    patterns = payload.get("patterns")
    if not isinstance(cells, dict):
        errors.append(_issue(path_text, "index_missing_cells", "Pattern index requires a cells object"))
    if not isinstance(patterns, dict):
        errors.append(_issue(path_text, "index_missing_patterns", "Pattern index requires a patterns object"))
    if errors:
        return errors, warnings
    for cell, pattern_ids in cells.items():  # type: ignore
        if not isinstance(pattern_ids, list):
            errors.append(_issue(path_text, "index_cell_not_list", f"Cell {cell} must contain a list of pattern ids"))
            continue
        for pattern_id in pattern_ids:
            if pattern_id not in patterns:  # type: ignore
                errors.append(_issue(path_text, "index_dangling_pattern_id", f"Cell {cell} references missing pattern {pattern_id}"))
    for pattern_id, pattern in patterns.items():  # type: ignore
        pattern_errors, pattern_warnings = validate_pattern_object_payload(pattern, path=f"{path_text}#{pattern_id}")
        errors.extend(pattern_errors)
        warnings.extend(pattern_warnings)
    return errors, warnings


def validate_pattern_object_payload(
    payload: Any,
    *,
    path: Path | str = "<pattern>",
) -> tuple[list[PatternValidationIssue], list[PatternValidationIssue]]:
    path_text = str(path)
    errors: list[PatternValidationIssue] = []
    warnings: list[PatternValidationIssue] = []
    if not isinstance(payload, dict):
        return [_issue(path_text, "pattern_must_be_object", "Pattern file must be a JSON object")], warnings

    required = {"id", "mode", "difficulty", "length_beats", "objects", "entry", "exit", "tested", "source"}
    missing = required - set(payload)
    if missing:
        errors.append(_issue(path_text, "pattern_missing_required_fields", f"Missing required fields: {', '.join(sorted(missing))}"))
    mode = str(payload.get("mode", ""))
    difficulty = str(payload.get("difficulty", ""))
    if mode not in VALID_GAME_MODES:
        errors.append(_issue(path_text, "pattern_unknown_mode", f"Unknown mode: {mode}"))
    if difficulty not in VALID_DIFFICULTIES:
        errors.append(_issue(path_text, "pattern_unknown_difficulty", f"Unknown difficulty: {difficulty}"))
    try:
        length_beats = float(payload.get("length_beats"))  # type: ignore
        if length_beats <= 0 or length_beats > 128:
            errors.append(_issue(path_text, "pattern_length_out_of_range", "length_beats must be in (0, 128]"))
    except (TypeError, ValueError):
        length_beats = 0.0
        errors.append(_issue(path_text, "pattern_length_invalid", "length_beats must be numeric"))

    objects = payload.get("objects")
    if not isinstance(objects, list):
        errors.append(_issue(path_text, "pattern_objects_must_be_list", "objects must be a list"))
        return errors, warnings
    if not objects:
        errors.append(_issue(path_text, "pattern_object_count_zero", "Pattern objects must not be empty"))

    declared_groups = _string_set(payload.get("group_symbols", []))
    declared_colors = _string_set(payload.get("color_symbols", []))
    for index, obj in enumerate(objects):
        if not isinstance(obj, dict):
            errors.append(_issue(path_text, "pattern_object_must_be_object", f"objects[{index}] must be an object"))
            continue
        _validate_pattern_object(
            obj,
            index=index,
            path=path_text,
            length_beats=length_beats,
            declared_groups=declared_groups,
            declared_colors=declared_colors,
            errors=errors,
            warnings=warnings,
        )
    return errors, warnings


def _validate_pattern_object(
    obj: dict[str, Any],
    *,
    index: int,
    path: str,
    length_beats: float,
    declared_groups: set[str],
    declared_colors: set[str],
    errors: list[PatternValidationIssue],
    warnings: list[PatternValidationIssue],
) -> None:
    label = f"objects[{index}]"
    for field_name in ("id", "x_beat", "y", "role"):
        if field_name not in obj:
            errors.append(_issue(path, "pattern_object_missing_field", f"{label} missing {field_name}"))
    for field_name, minimum, maximum in (
        ("x_beat", -1.0, max(1.0, length_beats) + 1.0),
        ("y", -300.0, 3000.0),
        ("dx", -2000.0, 2000.0),
        ("dy", -2000.0, 2000.0),
    ):
        if field_name in obj:
            _validate_numeric_range(obj[field_name], minimum, maximum, path, f"{label}.{field_name}", errors)

    role = str(obj.get("role", ""))
    if role == "trigger":
        trigger_type = str(obj.get("trigger_type", ""))
        if trigger_type not in ALLOWED_TRIGGER_TYPES:
            errors.append(_issue(path, "pattern_trigger_type_not_allowed", f"{label} has unsupported trigger type {trigger_type}"))

    for key in ("group_id", "group_ids", "target_group", "color_channel", "color_channel_id"):
        if key in obj:
            warnings.append(_issue(path, "pattern_uses_concrete_allocator_id", f"{label}.{key} should be symbolic in source data"))

    group_symbol = obj.get("group_symbol")
    if isinstance(group_symbol, str):
        declared_groups.add(group_symbol)
    for key in ("target_group_symbol", "trigger_group_symbol"):
        value = obj.get(key)
        if isinstance(value, str) and value not in declared_groups:
            errors.append(_issue(path, "pattern_dangling_group_symbol", f"{label}.{key} references undefined group symbol {value}"))

    color_symbol = obj.get("color_symbol")
    if isinstance(color_symbol, str):
        declared_colors.add(color_symbol)
    target_color = obj.get("target_color_symbol")
    if isinstance(target_color, str) and target_color not in declared_colors:
        errors.append(_issue(path, "pattern_dangling_color_symbol", f"{label}.target_color_symbol references undefined color symbol {target_color}"))


def _validate_numeric_range(
    value: Any,
    minimum: float,
    maximum: float,
    path: str,
    label: str,
    errors: list[PatternValidationIssue],
) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        errors.append(_issue(path, "pattern_numeric_field_invalid", f"{label} must be numeric"))
        return
    if numeric < minimum or numeric > maximum:
        errors.append(_issue(path, "pattern_numeric_field_out_of_range", f"{label} is outside [{minimum}, {maximum}]"))


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item) for item in value if isinstance(item, str) and item}


def _issue(path: str, code: str, message: str) -> PatternValidationIssue:
    return PatternValidationIssue(path=path, code=code, message=message)
