# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
from pathlib import Path

from gmdgen.ai.dataset_index import (
    build_dataset_index,
    build_generation_context_from_dataset,
    dataset_cache_path,
    ensure_dataset_dir,
    invalidate_dataset_cache_if_changed,
    load_dataset_document,
    load_dataset_index,
    resolve_dataset_dir,
    save_dataset_index,
    scan_dataset_dir,
    validate_dataset_dir,
)
from gmdgen.ai.training import AutoTrainingConfig, run_auto_training
from gmdgen.generate.audio_conditioned import _append_learning_context
from gmdgen.learning.dataset_memory import (
    dataset_learning_examples_dir,
    load_dataset_learning_examples,
    rebuild_index_after_learning_save,
    save_learning_example_to_dataset,
)


def test_default_dataset_dir_is_dataset() -> None:
    path = resolve_dataset_dir()

    assert path.name == "dataset"
    assert path.parent.name == "gmdgen"


def test_resolve_dataset_dir_relative_to_project_root(tmp_path: Path) -> None:
    path = resolve_dataset_dir("my data", project_root=tmp_path)

    assert path == (tmp_path / "my data").resolve()


def test_dataset_dir_created_if_missing(tmp_path: Path) -> None:
    path = ensure_dataset_dir("dataset", project_root=tmp_path)
    status = validate_dataset_dir(path)

    assert path.exists()
    assert status.exists is True
    assert status.is_dir is True


def test_dataset_dir_with_spaces(tmp_path: Path) -> None:
    path = ensure_dataset_dir("data with spaces", project_root=tmp_path)

    assert path.exists()
    assert " " in str(path)


def test_dataset_dir_with_korean_path_if_possible(tmp_path: Path) -> None:
    path = ensure_dataset_dir("데이터셋", project_root=tmp_path)
    (path / "설명.md").write_text("trigger schema", encoding="utf-8")

    result = scan_dataset_dir(path)

    assert result.files_indexed == 1


def test_scan_dataset_dir_recursive(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    nested = dataset / "docs" / "nested"
    nested.mkdir(parents=True)
    (nested / "guide.md").write_text("Geometry Dash trigger schema", encoding="utf-8")
    (dataset / "ref.gmd").write_text("kA11,0;1,1,2,30,3,90;", encoding="utf-8")

    result = scan_dataset_dir(dataset)

    assert result.total_files_found == 2
    assert result.files_indexed == 2
    assert result.indexed_extensions[".gmd"] == 1


def test_dataset_loader_supports_md_txt_json_gmd(tmp_path: Path) -> None:
    dataset = ensure_dataset_dir(tmp_path / "dataset")
    (dataset / "a.md").write_text("alpha", encoding="utf-8")
    (dataset / "b.txt").write_text("beta", encoding="utf-8")
    (dataset / "c.json").write_text(json.dumps({"validation_report": {"score": 1}}), encoding="utf-8")
    (dataset / "d.gmd").write_text("kA11,0;1,1,2,30,3,90;", encoding="utf-8")

    docs = [load_dataset_document(path, dataset_dir=dataset) for path in sorted(dataset.iterdir())]

    assert {doc.extension for doc in docs} == {".md", ".txt", ".json", ".gmd"}
    assert all(doc.chunks for doc in docs)


def test_dataset_loader_skips_unknown_extension(tmp_path: Path) -> None:
    dataset = ensure_dataset_dir(tmp_path / "dataset")
    (dataset / "ignore.bin").write_bytes(b"\x00\x01")

    result = scan_dataset_dir(dataset)

    assert result.files_skipped == 1
    assert result.skipped_extensions[".bin"] == 1


def test_dataset_loader_continues_on_bad_file(tmp_path: Path) -> None:
    dataset = ensure_dataset_dir(tmp_path / "dataset")
    (dataset / "good.md").write_text("good", encoding="utf-8")
    (dataset / "bad.json").write_text("{not json", encoding="utf-8")

    index = build_dataset_index(dataset)

    assert len(index.documents) == 2
    assert index.scan_result["files_indexed"] == 2


def test_dataset_index_contains_all_supported_files(tmp_path: Path) -> None:
    dataset = ensure_dataset_dir(tmp_path / "dataset")
    for name in ("a.md", "b.txt", "c.jsonl", "d.py", "e.csv", "f.yaml", "g.yml"):
        (dataset / name).write_text("trigger schema\n", encoding="utf-8")

    index = build_dataset_index(dataset)

    assert len(index.documents) == 7
    assert len(index.chunks) >= 7


def test_dataset_index_cache_invalidates_when_file_changes(tmp_path: Path) -> None:
    dataset = ensure_dataset_dir(tmp_path / "dataset")
    file_path = dataset / "a.md"
    file_path.write_text("one", encoding="utf-8")
    index = build_dataset_index(dataset)
    cache = save_dataset_index(index, dataset_cache_path(dataset))

    assert load_dataset_index(cache) is not None
    assert invalidate_dataset_cache_if_changed(dataset, cache) is False
    file_path.write_text("two", encoding="utf-8")
    assert invalidate_dataset_cache_if_changed(dataset, cache) is True


def test_app_startup_uses_dataset_dir(tmp_path: Path) -> None:
    dataset = ensure_dataset_dir(tmp_path / "dataset")
    (dataset / "guide.md").write_text("playability rules", encoding="utf-8")

    result, _index = run_auto_training(
        AutoTrainingConfig(
            dataset_dir=str(dataset),
            context_dirs=[],
            reference_level_dirs=[],
            schema_paths=[],
            cache_dir=str(tmp_path / "cache"),
        )
    )

    assert result.dataset_dir == str(dataset)
    assert result.dataset_document_count == 1


def test_rebuild_context_indexes_entire_dataset(tmp_path: Path) -> None:
    dataset = ensure_dataset_dir(tmp_path / "dataset")
    (dataset / "a.md").write_text("a", encoding="utf-8")
    (dataset / "nested").mkdir()
    (dataset / "nested" / "b.txt").write_text("b", encoding="utf-8")

    result, _index = run_auto_training(
        AutoTrainingConfig(
            dataset_dir=str(dataset),
            context_dirs=[],
            reference_level_dirs=[],
            schema_paths=[],
            cache_dir=str(tmp_path / "cache"),
            rebuild=True,
        )
    )

    assert result.dataset_document_count == 2
    assert result.dataset_chunk_count >= 2


def test_generation_context_uses_dataset_index(tmp_path: Path) -> None:
    dataset = ensure_dataset_dir(tmp_path / "dataset")
    (dataset / "style.md").write_text("drop section pulse trigger cyber style", encoding="utf-8")
    index = build_dataset_index(dataset)

    context = build_generation_context_from_dataset(index, {"prompt": "cyber drop"})

    assert "cyber" in context.relevant_docs_summary
    assert context.dataset_stats_summary["document_count"] == 1


def test_generation_context_respects_max_chars(tmp_path: Path) -> None:
    dataset = ensure_dataset_dir(tmp_path / "dataset")
    (dataset / "big.md").write_text("drop " * 2000, encoding="utf-8")
    index = build_dataset_index(dataset)

    context = build_generation_context_from_dataset(index, {"prompt": "drop"}, max_chars=300)

    assert len(context.relevant_docs_summary) <= 320


def test_prompt_includes_dataset_context_summary(tmp_path: Path) -> None:
    dataset = ensure_dataset_dir(tmp_path / "dataset")
    (dataset / "guide.md").write_text("dataset prompt context", encoding="utf-8")
    save_dataset_index(build_dataset_index(dataset), dataset_cache_path(dataset))

    chunks = _append_learning_context(
        [{"path": "base", "title": "Base", "text": "base"}],
        {"dataset_dir": str(dataset), "use_dataset_context": True, "ollama_max_context_chars": 2000},
    )

    assert any(chunk.get("path") == "dataset_context" for chunk in chunks)


def test_dataset_context_does_not_dump_entire_files(tmp_path: Path) -> None:
    dataset = ensure_dataset_dir(tmp_path / "dataset")
    raw = "secretish-long-level-string " * 1000
    (dataset / "long.md").write_text(raw, encoding="utf-8")
    index = build_dataset_index(dataset, max_total_context_chars=1000)

    context = build_generation_context_from_dataset(index, {"prompt": "level"}, max_chars=500)

    assert len(context.relevant_docs_summary) < len(raw)


def test_learning_example_saved_under_dataset(tmp_path: Path) -> None:
    dataset = ensure_dataset_dir(tmp_path / "dataset")
    example_id = save_learning_example_to_dataset({"example_id": "ex", "ollama_base_url": "sk-secret"}, dataset)
    records = load_dataset_learning_examples(dataset)

    assert example_id == "ex"
    assert records[0]["example_id"] == "ex"
    assert "sk-secret" not in (dataset_learning_examples_dir(dataset) / "examples.jsonl").read_text(encoding="utf-8")


def test_dataset_index_updates_after_learning_save(tmp_path: Path) -> None:
    dataset = ensure_dataset_dir(tmp_path / "dataset")
    save_learning_example_to_dataset({"example_id": "ex", "user_rating": 5}, dataset)

    index = rebuild_index_after_learning_save(dataset)

    assert index["scan_result"]["files_indexed"] >= 1
