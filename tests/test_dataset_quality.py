from __future__ import annotations

from pathlib import Path

from gmdgen.eval.dataset_quality import scan_dataset_quality


def test_dataset_quality_detects_usable_files(tmp_path: Path) -> None:
    (tmp_path / "test.gmd").write_text("kA11,0;", encoding="utf-8")
    (tmp_path / "notes.md").write_text("notes", encoding="utf-8")
    (tmp_path / "data.jsonl").write_text("{}", encoding="utf-8")
    
    report = scan_dataset_quality(tmp_path)
    assert report.total_files == 3
    assert report.usable_reference_levels == 1
    assert report.usable_docs == 1
    assert report.usable_learning_examples == 1
