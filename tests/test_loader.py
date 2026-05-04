# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from pathlib import Path

from gmdgen.data.loader import load_dataset_with_report
from gmdgen.io.gmd_decoder import encode_level_data
from gmdgen.io.gmd_writer import write_gmd_file


def test_loader_reports_missing_k4_files(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    write_gmd_file(
        dataset_dir / "valid.gmd",
        {
            "k2": ("s", "valid"),
            "k4": ("s", encode_level_data("1,1,2,30,3,180;")),
        },
    )
    write_gmd_file(
        dataset_dir / "meta_only.gmd",
        {
            "k2": ("s", "meta_only"),
        },
    )

    result = load_dataset_with_report(dataset_dir)

    assert len(result.records) == 1
    assert result.report.files_scanned == 2
    assert result.report.loaded_records == 1
    assert result.report.skipped_missing_k4 == 1
    assert result.report.skipped_total == 1
