from __future__ import annotations

"""Tests for the learning data pipeline: dataset feature extraction and
style-profile construction."""

from pathlib import Path

import pytest

from gmdgen.learning.feature_extractor import (
    LearnedDataStore,
    LevelFeatureSummary,
    build_palette_from_learned_store,
    build_style_profile,
    extract_level_features,
    palette_metrics,
)


SYNTHETIC_LEVEL = ";".join(
    f"1,{obj_id},2,{x},3,30"
    for obj_id, x in [
        ("1", 0), ("1", 30), ("1", 60),
        ("36", 90), ("141", 120), ("211", 150),
        ("503", 180), ("504", 210), ("1734", 240),
        ("1736", 270), ("259", 300), ("266", 330),
    ]
) + ";"


def test_extract_level_features_summarizes_objects():
    summary = extract_level_features(SYNTHETIC_LEVEL, source_name="synthetic")
    assert isinstance(summary, LevelFeatureSummary)
    assert summary.object_count >= 1


def test_build_style_profile_aggregates_summaries():
    summary = extract_level_features(SYNTHETIC_LEVEL, source_name="s1")
    profile = build_style_profile([summary])
    assert profile.source_count == 1
    # Common object IDs should include some of our synthetic IDs
    assert any(oid in profile.common_object_ids for oid in ["1", "36", "211", "503"])


def test_palette_from_empty_store_returns_empty_dict():
    palette = build_palette_from_learned_store(None)
    assert palette == {}
    palette = build_palette_from_learned_store(LearnedDataStore())
    assert palette == {}


def test_palette_from_populated_store_groups_by_class():
    store = LearnedDataStore(
        object_distributions={
            "1": 100, "2": 50, "211": 80, "503": 30,
            "36": 40, "141": 20, "1734": 25,
        }
    )
    palette = build_palette_from_learned_store(store)
    assert palette, "Expected non-empty palette from populated store"
    metrics = palette_metrics(palette)
    assert metrics["unique_object_id_count"] >= 5


def test_palette_min_occurrences_filter():
    store = LearnedDataStore(object_distributions={"1": 100, "211": 1, "999": 5})
    palette = build_palette_from_learned_store(store, min_occurrences=10)
    flat = [oid for ids in palette.values() for oid in ids]
    assert "211" not in flat
    assert "999" not in flat
    assert "1" in flat


def test_palette_max_per_class_truncates():
    distribution = {str(1000 + i): 50 - i for i in range(50)}
    store = LearnedDataStore(object_distributions=distribution)
    palette = build_palette_from_learned_store(store, max_per_class=10)
    for cls, ids in palette.items():
        assert len(ids) <= 10, f"Class {cls} has {len(ids)} ids, exceeds max_per_class=10"


def test_dataset_directory_not_committed_to_repo():
    """User-supplied dataset/*.gmd files should not be tracked by git."""
    import subprocess
    repo_root = Path(__file__).parent.parent
    result = subprocess.run(
        ["git", "ls-files", "dataset/"],
        capture_output=True, text=True, cwd=str(repo_root),
    )
    tracked = [line for line in result.stdout.splitlines() if line.strip().endswith(".gmd")]
    # Allow tests/fixtures/levels/*.gmd in tests, but no top-level dataset/*.gmd.
    assert not tracked, f"User dataset .gmd files are tracked by git: {tracked[:5]}"


def test_palette_metrics_returns_serializable_dict():
    palette = {"DECORATION": ["211", "503"], "STRUCTURE": ["1", "2"]}
    metrics = palette_metrics(palette)
    import json
    json.dumps(metrics)  # must be JSON-serializable
    assert metrics["class_count"] == 2
    assert metrics["unique_object_id_count"] == 4
