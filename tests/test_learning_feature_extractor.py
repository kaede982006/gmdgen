from __future__ import annotations

import json
from pathlib import Path

from gmdgen.learning.feature_extractor import (
    clear_learned_data_store,
    export_finetune_jsonl_from_learning_store,
    export_preference_pairs_from_learning_store,
    extract_level_features,
    learn_from_directory,
    learn_from_file,
    learned_data_store_path,
    load_learned_data_store,
    retrieve_motifs_for_section,
    save_learned_data_store,
    summarize_learned_data_for_prompt,
    update_learned_data_store,
)


def _level() -> str:
    return (
        "kA11,0;"
        "1,1,2,0,3,90,57,1;"
        "1,500,2,60,3,240;"
        "1,36,2,120,3,180;"
        "1,35,2,180,3,132;"
        "1,1006,2,180,3,300,51,1;"
        "1,1,2,240,3,90;"
        "1,500,2,300,3,240;"
    )


def test_learn_from_gmd_file_extracts_features(tmp_path: Path) -> None:
    path = tmp_path / "ref.gmd"
    path.write_text(_level(), encoding="utf-8")

    store = learn_from_file(path)

    assert len(store.learned_levels) == 1
    assert store.learned_levels[0]["object_count"] > 0
    assert store.object_distributions["1"] >= 1


def test_learn_from_directory_indexes_multiple_levels(tmp_path: Path) -> None:
    (tmp_path / "a.gmd").write_text(_level(), encoding="utf-8")
    (tmp_path / "b.txt").write_text(_level(), encoding="utf-8")

    store = learn_from_directory(tmp_path)

    assert len(store.learned_levels) == 2
    assert len(store.motif_bank) >= 1


def test_motif_bank_extracts_motifs(tmp_path: Path) -> None:
    path = tmp_path / "motif.gmd"
    path.write_text(_level(), encoding="utf-8")

    store = learn_from_file(path)

    assert store.motif_bank
    assert store.motif_bank[0]["object_ids"]


def test_style_profile_created_from_learned_levels(tmp_path: Path) -> None:
    path = tmp_path / "style.gmd"
    path.write_text(_level(), encoding="utf-8")

    store = learn_from_file(path)

    assert store.style_profiles
    assert store.style_profiles[0]["source_count"] == 1
    assert store.style_profiles[0]["common_object_ids"]


def test_learned_data_store_save_load(tmp_path: Path) -> None:
    path = tmp_path / "store.gmd"
    path.write_text(_level(), encoding="utf-8")
    store = learn_from_file(path)

    save_path = save_learned_data_store(store, store_dir=tmp_path / "learned")
    loaded = load_learned_data_store(store_dir=tmp_path / "learned")

    assert save_path.exists()
    assert len(loaded.learned_levels) == 1


def test_clear_learned_data_store(tmp_path: Path) -> None:
    save_learned_data_store(learn_from_file(_write_level(tmp_path)), store_dir=tmp_path / "learned")

    assert clear_learned_data_store(store_dir=tmp_path / "learned") is True
    assert not learned_data_store_path(tmp_path / "learned").exists()


def test_learning_does_not_store_api_key(tmp_path: Path) -> None:
    store = learn_from_file(_write_level(tmp_path))
    store.feedback_examples.append({"ollama_base_url": "sk-secret"})

    save_learned_data_store(store, store_dir=tmp_path / "learned")
    text = learned_data_store_path(tmp_path / "learned").read_text(encoding="utf-8")

    assert "sk-secret" not in text
    assert "ollama_base_url" not in text


def test_learning_handles_invalid_gmd_gracefully(tmp_path: Path) -> None:
    path = tmp_path / "bad.gmd"
    path.write_text("not a gd level", encoding="utf-8")

    store = learn_from_file(path)

    assert store.learned_levels == []
    assert store.failure_patterns or store.success_patterns == []


def test_learned_data_prompt_summary_and_motif_retrieval(tmp_path: Path) -> None:
    store = learn_from_file(_write_level(tmp_path))
    save_learned_data_store(store, store_dir=tmp_path / "learned")

    summary = summarize_learned_data_for_prompt(store_dir=tmp_path / "learned")
    motifs = retrieve_motifs_for_section({"section_type": "drop"}, store)

    assert summary["learned_level_count"] == 1
    assert "learned_object_distribution" in summary
    assert isinstance(motifs, list)


def test_prompt_does_not_include_full_raw_level_string(tmp_path: Path) -> None:
    store = learn_from_file(_write_level(tmp_path))
    summary_text = json.dumps(summarize_learned_data_for_prompt(store), ensure_ascii=False)

    assert "kA11,0;1,1,2,0" not in summary_text


def test_update_learned_data_store_merges_distributions(tmp_path: Path) -> None:
    one = learn_from_file(_write_level(tmp_path, name="one.gmd"))
    two = learn_from_file(_write_level(tmp_path, name="two.gmd"))

    merged = update_learned_data_store(one, two)

    assert len(merged.learned_levels) == 2
    assert int(merged.object_distributions["1"]) >= 2


def test_export_finetune_jsonl_from_learning_store(tmp_path: Path) -> None:
    store = learn_from_file(_write_level(tmp_path))
    store.learned_levels[0]["user_rating"] = 5
    save_learned_data_store(store, store_dir=tmp_path / "learned")
    output = tmp_path / "export.jsonl"

    export_finetune_jsonl_from_learning_store(output, store_dir=tmp_path / "learned")

    assert output.exists()
    assert "sk-" not in output.read_text(encoding="utf-8")


def test_export_preference_pairs(tmp_path: Path) -> None:
    store = learn_from_file(_write_level(tmp_path))
    good = dict(store.learned_levels[0])
    good["source_name"] = "good"
    good["user_rating"] = 5
    bad = dict(store.learned_levels[0])
    bad["source_name"] = "bad"
    bad["user_rating"] = 1
    store.learned_levels = [good, bad]
    save_learned_data_store(store, store_dir=tmp_path / "learned")
    output = tmp_path / "pref.jsonl"

    export_preference_pairs_from_learning_store(output, store_dir=tmp_path / "learned")

    assert "chosen" in output.read_text(encoding="utf-8")


def test_extract_level_features_from_validation_json(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"level_string": _level()}), encoding="utf-8")

    summary = extract_level_features(path)

    assert summary.object_count > 0


def _write_level(tmp_path: Path, *, name: str = "level.gmd") -> Path:
    path = tmp_path / name
    path.write_text(_level(), encoding="utf-8")
    return path
