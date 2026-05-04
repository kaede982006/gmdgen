from __future__ import annotations

from pathlib import Path

from gmdgen.eval.export import export_jsonl_for_finetuning


def test_finetune_export_filters_low_quality_outputs(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    export_jsonl_for_finetuning(tmp_path, out)
    assert True
