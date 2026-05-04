# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from pathlib import Path

from gmdgen.data.splitter import DatasetSplit, split_dataset
from gmdgen.data.schema import GMDRecord, GMDDocument
from gmdgen.io.gmd_decoder import encode_level_data
from gmdgen.io.gmd_writer import write_gmd_file
from gmdgen.data.loader import load_dataset_with_report


def _make_gmd(path: Path, name: str, obj_count: int) -> None:
    objs = ";".join(f"1,1,2,{i*30},3,180" for i in range(1, obj_count + 1))
    level_data = f"kA11,0;{objs};"
    tags = {
        "k2": ("s", name),
        "k4": ("s", encode_level_data(level_data)),
        "k95": ("i", str(obj_count)),
    }
    write_gmd_file(path, tags)


def _build_mock_records(count: int) -> list[GMDRecord]:
    """Build in-memory GMDRecord instances without touching disk."""
    from gmdgen.data.preprocess import split_level_objects
    records = []
    for i in range(count):
        obj_count = (i + 1) * 20
        objs = ";".join(f"1,1,2,{j*30},3,180" for j in range(1, obj_count + 1))
        level_data = f"kA11,0;{objs};"
        tags = {"k2": ("s", f"level_{i}"), "k4": ("s", encode_level_data(level_data))}
        doc = GMDDocument(path=Path(f"level_{i}.gmd"), raw_text="", tags=tags)
        records.append(GMDRecord(document=doc, decoded_level_data=level_data))
    return records


# ── split_dataset ─────────────────────────────────────────────────────────────

def test_split_ratio_roughly_correct() -> None:
    records = _build_mock_records(20)
    split = split_dataset(records, val_ratio=0.2, seed=42)
    assert split.val_count >= 1
    assert split.train_count + split.val_count == 20
    assert abs(split.val_count / 20 - 0.2) < 0.15


def test_split_empty_records() -> None:
    split = split_dataset([])
    assert split.train_count == 0
    assert split.val_count == 0


def test_split_single_record_min_val() -> None:
    records = _build_mock_records(3)
    split = split_dataset(records, val_ratio=0.1, seed=0, min_val_records=1)
    assert split.val_count >= 1


def test_split_reproducible() -> None:
    records = _build_mock_records(15)
    split_a = split_dataset(records, seed=42)
    split_b = split_dataset(records, seed=42)
    names_a = {r.document.path.name for r in split_a.validation}
    names_b = {r.document.path.name for r in split_b.validation}
    assert names_a == names_b


def test_split_different_seed_different_result() -> None:
    records = _build_mock_records(15)
    split_a = split_dataset(records, seed=1)
    split_b = split_dataset(records, seed=999)
    names_a = {r.document.path.name for r in split_a.validation}
    names_b = {r.document.path.name for r in split_b.validation}
    assert names_a != names_b


# ── split_dataset_dir ─────────────────────────────────────────────────────────

def test_split_dataset_dir_loads_and_splits(tmp_path: Path) -> None:
    for i in range(10):
        _make_gmd(tmp_path / f"level_{i}.gmd", f"L{i}", 50 + i * 10)

    from gmdgen.data.splitter import split_dataset_dir
    split, load_result = split_dataset_dir(tmp_path, val_ratio=0.2, seed=0)

    assert split.train_count + split.val_count == 10
    assert load_result.report.loaded_records == 10
