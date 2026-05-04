# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DatasetQualityReport:
    dataset_dir: str = ""
    total_files: int = 0
    usable_docs: int = 0
    usable_reference_levels: int = 0
    usable_learning_examples: int = 0
    rejected_files: int = 0
    low_quality_examples: int = 0
    corrupted_records: int = 0
    duplicate_records: int = 0
    prompt_leak_records: int = 0
    garbage_description_records: int = 0
    missing_metadata_records: int = 0
    suspicious_large_files: int = 0
    recommended_cleanup_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_dir": self.dataset_dir,
            "total_files": self.total_files,
            "usable_docs": self.usable_docs,
            "usable_reference_levels": self.usable_reference_levels,
            "usable_learning_examples": self.usable_learning_examples,
            "rejected_files": self.rejected_files,
            "low_quality_examples": self.low_quality_examples,
            "corrupted_records": self.corrupted_records,
            "duplicate_records": self.duplicate_records,
            "prompt_leak_records": self.prompt_leak_records,
            "garbage_description_records": self.garbage_description_records,
            "missing_metadata_records": self.missing_metadata_records,
            "suspicious_large_files": self.suspicious_large_files,
            "recommended_cleanup_actions": list(self.recommended_cleanup_actions),
        }


def scan_dataset_quality(dataset_dir: Path) -> DatasetQualityReport:
    report = DatasetQualityReport(dataset_dir=str(dataset_dir))
    if not dataset_dir.exists():
        return report

    for path in dataset_dir.rglob("*"):
        if path.is_file():
            report.total_files += 1
            if path.suffix == ".gmd":
                if path.stat().st_size > 5_000_000:
                    report.suspicious_large_files += 1
                report.usable_reference_levels += 1
            elif path.suffix in {".md", ".txt"}:
                report.usable_docs += 1
            elif path.suffix == ".jsonl":
                report.usable_learning_examples += 1

    return report


def quarantine_bad_dataset_records(dataset_dir: Path, report: DatasetQualityReport) -> None:
    pass


def rebuild_clean_dataset_index(dataset_dir: Path) -> None:
    pass


def export_clean_dataset_manifest(dataset_dir: Path, output_path: Path) -> None:
    pass
