# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gmdgen.data.preprocess import split_level_objects
from gmdgen.io.gmd_decoder import decode_level_data
from gmdgen.io.gmd_parser import parse_gmd_file


@dataclass(slots=True)
class DatasetCleaningReport:
    checked_files: int = 0
    valid_files: int = 0
    invalid_files: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    repair_suggestions: list[str] = field(default_factory=list)
    destructive_changes: bool = False

    @property
    def passed(self) -> bool:
        return not self.invalid_files

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_files": self.checked_files,
            "valid_files": self.valid_files,
            "invalid_files": list(self.invalid_files),
            "warnings": list(self.warnings),
            "repair_suggestions": list(self.repair_suggestions),
            "destructive_changes": self.destructive_changes,
            "passed": self.passed,
        }


def inspect_dataset_gmd_files(root: Path | str) -> DatasetCleaningReport:
    """Inspect user .gmd files without deleting or mutating them."""
    directory = Path(root)
    report = DatasetCleaningReport()
    for path in sorted(directory.rglob("*.gmd")) if directory.exists() else []:
        report.checked_files += 1
        try:
            document = parse_gmd_file(path)
            k4 = document.tags.get("k4")
            if not k4:
                raise ValueError("missing_k4_level_data")
            decoded = decode_level_data(k4[1])
            objects = split_level_objects(decoded)
        except Exception as exc:  # noqa: BLE001
            report.invalid_files.append({"path": str(path), "error": str(exc)})
            report.repair_suggestions.append(f"Quarantine or manually repair {path}")
            continue
        if not objects:
            report.invalid_files.append({"path": str(path), "error": "object_count_zero"})
            report.repair_suggestions.append(f"Review empty level data in {path}")
            continue
        report.valid_files += 1
        k95 = document.tags.get("k95")
        if k95:
            try:
                declared = int(k95[1])
                if declared != len(objects):
                    report.warnings.append(f"{path}: k95={declared} actual_objects={len(objects)}")
            except (TypeError, ValueError):
                report.warnings.append(f"{path}: k95 is not an integer")
    return report
