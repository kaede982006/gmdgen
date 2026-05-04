from __future__ import annotations

import json
from pathlib import Path

from gmdgen.io.gmd_decoder import encode_level_data
from gmdgen.io.gmd_writer import write_gmd_file
from gmdgen.train.trainer import train_from_config


def _write_sample_gmd(path: Path, name: str, object_ids: list[str]) -> None:
    chunks = []
    x_pos = 0
    for object_id in object_ids:
        x_pos += 30
        chunks.append(f"1,{object_id},2,{x_pos},3,180")
    level_data = "kA11,0;" + ";".join(chunks) + ";"

    tags = {
        "k2": ("s", name),
        "k4": ("s", encode_level_data(level_data)),
        "k95": ("i", str(len(object_ids))),
    }
    write_gmd_file(path, tags)


def test_train_creates_artifact(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    _write_sample_gmd(dataset_dir / "one.gmd", "one", ["1", "2", "3", "4"])
    _write_sample_gmd(dataset_dir / "two.gmd", "two", ["2", "3", "4", "5"])

    artifact_path = tmp_path / "artifacts" / "model.json"
    result = train_from_config(
        {
            "dataset_dir": str(dataset_dir),
            "artifact_path": str(artifact_path),
            "markov_order": 2,
            "min_objects_per_level": 1,
            "max_objects_per_level": 100,
            "seed": 123,
            "chunk_size": 2,
            "chunk_stride": 1,
            "max_chunks_per_level": 4,
            "min_chunk_objects": 1,
        }
    )

    assert artifact_path.exists()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert "model" in payload
    assert "generation_assets" in payload
    assert "object_prototypes" in payload["generation_assets"]
    assert payload["generation_assets"]["chunk_library"]
    assert payload["generation_assets"]["chunk_transition_counts"]
    assert result["used_records"] == 2
