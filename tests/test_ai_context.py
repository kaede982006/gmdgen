# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from pathlib import Path

from gmdgen.ai.context import (
    LocalKeywordRetriever,
    load_context_documents,
    summarize_context_documents,
    summarize_reference_level,
)


def test_load_context_documents_md_txt(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("Geometry Dash trigger schema", encoding="utf-8")
    (tmp_path / "rules.txt").write_text("time-X playability", encoding="utf-8")
    (tmp_path / "ignore.bin").write_bytes(b"no")

    docs = load_context_documents(tmp_path)

    assert {doc.title for doc in docs} == {"notes.md", "rules.txt"}


def test_context_summary_respects_max_chars(tmp_path: Path) -> None:
    (tmp_path / "long.md").write_text("a" * 1000, encoding="utf-8")
    docs = load_context_documents(tmp_path)

    summary = summarize_context_documents(docs, max_chars=100)

    assert sum(len(item["text"]) for item in summary) <= 100


def test_reference_level_summary_does_not_include_full_huge_string() -> None:
    level = "kA11,0;" + ";".join(f"1,1,2,{idx * 30},3,90" for idx in range(20)) + ";"
    summary = summarize_reference_level(level)

    assert summary["object_count"] >= 20
    assert summary["text_included"] is False
    assert "level" not in summary


def test_local_keyword_retriever_returns_relevant_chunks(tmp_path: Path) -> None:
    (tmp_path / "gd.md").write_text("Geometry Dash trigger target group schema", encoding="utf-8")
    (tmp_path / "other.md").write_text("unrelated", encoding="utf-8")
    docs = load_context_documents(tmp_path)
    retriever = LocalKeywordRetriever(docs)

    chunks = retriever.retrieve("trigger group", top_k=1)

    assert chunks
    assert chunks[0].title == "gd.md"
