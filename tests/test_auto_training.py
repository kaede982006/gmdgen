# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
from pathlib import Path

from gmdgen.ai.training import (
    AutoTrainingConfig,
    invalidate_cache_if_sources_changed,
    rebuild_auto_training_config,
    run_auto_training,
)


def test_auto_training_runs_without_ollama_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    docs = tmp_path / "docs"
    levels = tmp_path / "levels"
    docs.mkdir()
    levels.mkdir()
    (docs / "guide.md").write_text("trigger schema playability", encoding="utf-8")
    (levels / "ref.gmd").write_text("kA11,0;1,1,2,30,3,90;", encoding="utf-8")
    config = AutoTrainingConfig(
        context_dirs=[str(docs)],
        reference_level_dirs=[str(levels)],
        schema_paths=[],
        cache_dir=str(tmp_path / "cache"),
        max_context_chars=500,
    )
    result, index = run_auto_training(config)

    assert result.success is True
    assert result.document_count >= 1
    assert result.reference_level_count >= 1
    assert index.chunks


def test_auto_training_indexes_md_txt_json(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("alpha", encoding="utf-8")
    (docs / "b.txt").write_text("beta", encoding="utf-8")
    (docs / "c.json").write_text(json.dumps({"k": "v"}), encoding="utf-8")
    config = AutoTrainingConfig(
        context_dirs=[str(docs)],
        reference_level_dirs=[],
        schema_paths=[],
        cache_dir=str(tmp_path / "cache"),
    )
    result, _index = run_auto_training(config)
    assert result.document_count >= 3


def test_auto_training_summarizes_reference_gmd(tmp_path: Path) -> None:
    levels = tmp_path / "levels"
    levels.mkdir()
    (levels / "ref.gmd").write_text("kA11,0;1,1,2,30,3,90;1,8,2,60,3,90;", encoding="utf-8")
    config = AutoTrainingConfig(
        context_dirs=[],
        reference_level_dirs=[str(levels)],
        schema_paths=[],
        cache_dir=str(tmp_path / "cache"),
    )
    result, index = run_auto_training(config)
    assert result.reference_level_count == 1
    assert index.reference_levels[0]["object_count"] >= 2


def test_auto_training_cache_created(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("x", encoding="utf-8")
    config = AutoTrainingConfig(
        context_dirs=[str(docs)],
        reference_level_dirs=[],
        schema_paths=[],
        cache_dir=str(tmp_path / "cache"),
    )
    result, _index = run_auto_training(config)
    assert Path(result.cache_path).exists()


def test_auto_training_rebuild_invalidates_cache(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    file_path = docs / "a.md"
    file_path.write_text("x", encoding="utf-8")
    config = AutoTrainingConfig(
        context_dirs=[str(docs)],
        reference_level_dirs=[],
        schema_paths=[],
        cache_dir=str(tmp_path / "cache"),
    )
    run_auto_training(config)
    assert invalidate_cache_if_sources_changed(config) is False
    file_path.write_text("updated", encoding="utf-8")
    assert invalidate_cache_if_sources_changed(config) is True


def test_gui_startup_triggers_auto_training() -> None:
    import pytest
    pytest.importorskip('gmdgen.gui')
    from gmdgen.gui.app import GuiApplication

    app = GuiApplication()
    state = app.startup()
    assert state.training_result is not None
    assert state.context_ready == state.training_result.success


def test_context_rebuild_uses_safe_config_clone(tmp_path: Path) -> None:
    config = AutoTrainingConfig(context_dirs=[], reference_level_dirs=[], schema_paths=[], cache_dir=str(tmp_path / "cache"))

    rebuilt = rebuild_auto_training_config(config, rebuild=True)

    assert rebuilt.rebuild is True
    assert config.rebuild is False


def test_rebuild_context_no_attribute_error(tmp_path: Path) -> None:
    import pytest
    pytest.importorskip('gmdgen.gui')
    from gmdgen.gui.app import GuiApplication

    app = GuiApplication(
        training_config=AutoTrainingConfig(
            context_dirs=[],
            reference_level_dirs=[],
            schema_paths=[],
            cache_dir=str(tmp_path / "cache"),
        )
    )

    result = app.rebuild_context(rebuild=True)

    assert result.rebuild is True
    assert result.errors == []


def test_rebuild_context_failure_does_not_crash_gui(tmp_path: Path, monkeypatch) -> None:
    import pytest
    pytest.importorskip('gmdgen.gui')
    from gmdgen.gui.app import GuiApplication

    app = GuiApplication(
        training_config=AutoTrainingConfig(
            context_dirs=[],
            reference_level_dirs=[],
            schema_paths=[],
            cache_dir=str(tmp_path / "cache"),
        )
    )
    monkeypatch.setattr("gmdgen.gui.app.run_auto_training", lambda _config: (_ for _ in ()).throw(RuntimeError("boom")))

    result = app.rebuild_context(rebuild=True)

    assert result.success is False
    assert result.rebuild is True
    assert "context_rebuild_failed" in result.errors[0]


def test_auto_training_result_serializable(tmp_path: Path) -> None:
    config = AutoTrainingConfig(context_dirs=[], reference_level_dirs=[], schema_paths=[], cache_dir=str(tmp_path / "cache"))

    result, _index = run_auto_training(config)

    assert json.dumps(result.to_dict())
    assert "rebuild" in result.to_dict()


def test_auto_training_result_contains_dataset_stats(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "guide.md").write_text("dataset trigger schema", encoding="utf-8")
    config = AutoTrainingConfig(
        dataset_dir=str(dataset),
        context_dirs=[],
        reference_level_dirs=[],
        schema_paths=[],
        cache_dir=str(tmp_path / "cache"),
    )

    result, _index = run_auto_training(config)

    assert result.dataset_document_count == 1
    assert result.dataset_chunk_count >= 1
    assert result.dataset_scan_result["files_indexed"] == 1
