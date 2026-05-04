# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import logging
from pathlib import Path

from gmdgen.data.schema import DatasetLoadReport, DatasetLoadResult, GMDRecord
from gmdgen.io.gmd_decoder import decode_level_data
from gmdgen.io.gmd_parser import parse_gmd_file

LOGGER = logging.getLogger(__name__)


def load_dataset_with_report(
    dataset_dir: Path,
    *,
    log_skipped_files: bool = False,
) -> DatasetLoadResult:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    gmd_paths = sorted(dataset_dir.glob("*.gmd"))
    if not gmd_paths:
        raise ValueError(f"No .gmd files found in: {dataset_dir}")

    records: list[GMDRecord] = []
    skipped_missing_k4 = 0
    skipped_parse_failed = 0
    skipped_decode_failed = 0

    def _log_skip(message: str, *args: object) -> None:
        if log_skipped_files:
            LOGGER.warning(message, *args)
        else:
            LOGGER.debug(message, *args)

    for path in gmd_paths:
        try:
            document = parse_gmd_file(path)
        except Exception as exc:  # noqa: BLE001
            skipped_parse_failed += 1
            _log_skip("Skip %s: parse failed (%s)", path.name, exc)
            continue

        k4_entry = document.tags.get("k4")
        if not k4_entry:
            skipped_missing_k4 += 1
            _log_skip("Skip %s: missing k4 field", path.name)
            continue

        _, encoded_level_data = k4_entry
        try:
            decoded_level_data = decode_level_data(encoded_level_data)
        except Exception as exc:  # noqa: BLE001
            skipped_decode_failed += 1
            _log_skip("Skip %s: k4 decode failed (%s)", path.name, exc)
            continue

        records.append(GMDRecord(document=document, decoded_level_data=decoded_level_data))

    report = DatasetLoadReport(
        files_scanned=len(gmd_paths),
        loaded_records=len(records),
        skipped_missing_k4=skipped_missing_k4,
        skipped_parse_failed=skipped_parse_failed,
        skipped_decode_failed=skipped_decode_failed,
    )

    LOGGER.info(
        (
            "Dataset scanned: files=%d, loaded=%d, skipped_total=%d "
            "(missing_k4=%d, parse_failed=%d, decode_failed=%d)"
        ),
        report.files_scanned,
        report.loaded_records,
        report.skipped_total,
        report.skipped_missing_k4,
        report.skipped_parse_failed,
        report.skipped_decode_failed,
    )

    if not records:
        raise ValueError("No valid records left after parsing/decoding.")

    return DatasetLoadResult(records=records, report=report)


def load_dataset(dataset_dir: Path) -> list[GMDRecord]:
    return load_dataset_with_report(dataset_dir).records
