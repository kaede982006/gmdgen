# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from pathlib import Path

from gmdgen.generate.style_bank import build_motif_bank_from_files, extract_motifs_from_level, retrieve_motifs_for_section


def _level_string() -> str:
    objs = []
    for idx in range(18):
        object_id = "1006" if idx % 5 == 0 else "500" if idx % 3 == 0 else "1"
        objs.append(f"1,{object_id},2,{idx * 45},3,90")
    return ";".join(objs)


def test_motif_bank_extracts_motifs() -> None:
    motifs = extract_motifs_from_level(_level_string(), source_level="synthetic")

    assert motifs
    assert motifs[0].object_ids
    assert motifs[0].compact_plan_summary


def test_motif_retrieval_by_section_type(tmp_path: Path) -> None:
    path = tmp_path / "ref.gmd"
    path.write_text(_level_string(), encoding="utf-8")
    bank = build_motif_bank_from_files([path])

    retrieved = retrieve_motifs_for_section(bank, "normal", limit=3)

    assert len(retrieved) <= 3
    assert "motif_id" in retrieved[0]


def test_drop_section_retrieves_high_density_motifs(tmp_path: Path) -> None:
    path = tmp_path / "drop.gmd"
    path.write_text(_level_string(), encoding="utf-8")
    bank = build_motif_bank_from_files([path], window_x=360)

    motifs = bank.retrieve(section_type="drop", limit=2)

    assert motifs
    assert motifs[0].density >= motifs[-1].density


def test_prompt_includes_retrieved_motif_summaries(tmp_path: Path) -> None:
    path = tmp_path / "summary.gmd"
    path.write_text(_level_string(), encoding="utf-8")
    bank = build_motif_bank_from_files([path])

    prompt_items = bank.to_prompt_summaries(section_type="drop", limit=4)

    assert prompt_items
    assert "compact_plan_summary" in prompt_items[0]


def test_reference_level_full_string_not_dumped(tmp_path: Path) -> None:
    path = tmp_path / "safe.gmd"
    text = _level_string()
    path.write_text(text, encoding="utf-8")
    bank = build_motif_bank_from_files([path])

    summary_text = str(bank.to_prompt_summaries(section_type="drop"))

    assert text not in summary_text
