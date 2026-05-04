# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from pathlib import Path

from gmdgen.generate.generator import generate_from_config
from gmdgen.io.gmd_decoder import decode_level_description, encode_level_data
from gmdgen.io.gmd_parser import parse_gmd_file
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


def test_generate_creates_valid_gmd(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    _write_sample_gmd(dataset_dir / "one.gmd", "one", ["1", "2", "3", "4", "5"])
    _write_sample_gmd(dataset_dir / "two.gmd", "two", ["5", "4", "3", "2", "1"])

    artifact_path = tmp_path / "artifacts" / "model.json"
    train_from_config(
        {
            "dataset_dir": str(dataset_dir),
            "artifact_path": str(artifact_path),
            "markov_order": 2,
            "min_objects_per_level": 1,
            "max_objects_per_level": 100,
            "seed": 123,
        }
    )

    output_dir = tmp_path / "outputs"
    result = generate_from_config(
        {
            "artifact_path": str(artifact_path),
            "output_dir": str(output_dir),
            "output_name": "new_level",
            "num_objects": 20,
            "temperature": 1.0,
            "top_k": 5,
            "seed": 9,
            "prompt": "one",
            "prompt_strength": 0.8,
            "generation_passes": 2,
            "deliberation_width": 4,
        }
    )

    output_path = Path(result["output_path"])
    assert output_path.exists()
    assert result["valid"] is True
    assert result["prompt"] == "one"
    assert result["generation_passes"] == 2
    assert "selected_score" in result
    assert result["generated_description"] == "Generated Geometry Dash level."
    assert result["generation_mode"] == "style_only"
    assert result["style_submode"] in {"chunk_hybrid", "prototype_fallback", "template_edit"}

    document = parse_gmd_file(output_path)
    description = decode_level_description(document.tags["k3"][1])
    assert description == "Generated Geometry Dash level."
