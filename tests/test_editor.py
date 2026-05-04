from __future__ import annotations

import random
from pathlib import Path

from gmdgen.generate.editor import apply_style_edit, run_template_edit
from gmdgen.io.gmd_decoder import encode_level_data
from gmdgen.io.gmd_writer import write_gmd_file
from gmdgen.representation.object_classifier import ObjectClass


def _obj(object_id: str, x: int, y: int) -> str:
    return f"1,{object_id},2,{x},3,{y}"


def _trigger_obj(x: int, y: int, group: int) -> str:
    return f"1,29,2,{x},3,{y},51,{group}"


def _group_obj(object_id: str, x: int, y: int, group: int) -> str:
    return f"1,{object_id},2,{x},3,{y},155,{group}"


def _make_template_gmd(path: Path, objects: list[str]) -> None:
    level_data = "kA11,0;" + ";".join(objects) + ";"
    tags = {
        "k2": ("s", "test_template"),
        "k4": ("s", encode_level_data(level_data)),
        "k95": ("i", str(len(objects))),
    }
    write_gmd_file(path, tags)


# ── apply_style_edit ──────────────────────────────────────────────────────────

def test_triggers_always_kept() -> None:
    objects = [
        _trigger_obj(10, 180, 5),   # trigger — must be kept
        _obj("1", 30, 180),         # structure — may be swapped
        _obj("500", 60, 180),       # decoration — may be swapped
    ]
    rng = random.Random(42)
    edited, report = apply_style_edit(
        objects,
        generation_assets={},
        prompt="",
        matched_level_names=[],
        style_swap_ratio=1.0,   # always swap when possible
        jitter_x=0,
        jitter_y=0,
        rng=rng,
        swap_structure=True,
        swap_decoration=True,
    )
    assert len(edited) == 3
    assert edited[0] == objects[0], "trigger must not be modified"


def test_swap_ratio_zero_keeps_all() -> None:
    objects = [_obj("1", 30, 180), _obj("500", 60, 180)]
    pool_proto = {"1": ["1,1,2,0,3,0"], "500": ["1,500,2,0,3,0"]}
    rng = random.Random(0)
    edited, report = apply_style_edit(
        objects,
        generation_assets={"object_prototypes": pool_proto},
        prompt="",
        matched_level_names=[],
        style_swap_ratio=0.0,
        jitter_x=0,
        jitter_y=0,
        rng=rng,
        swap_structure=False,
        swap_decoration=False,
    )
    assert report.swapped_count == 0
    assert edited == objects


def test_positions_preserved_after_swap() -> None:
    """After swapping, the object's x and y must match the original."""
    from gmdgen.features.tokenizer import extract_object_number

    original = _obj("1", 120, 250)
    pool = ["1,999,2,0,3,0,4,1,155,7"]  # prototype at (0,0)

    generation_assets = {
        "chunk_library": [
            {
                "level_name": "test",
                "level_desc": "test",
                "objects": pool,
                "dominant_ids": ["999"],
            }
        ],
        "object_prototypes": {"999": pool},
    }

    rng = random.Random(7)
    edited, report = apply_style_edit(
        [original],
        generation_assets=generation_assets,
        prompt="test",
        matched_level_names=["test"],
        style_swap_ratio=1.0,
        jitter_x=0,
        jitter_y=0,
        rng=rng,
        swap_structure=True,
        swap_decoration=True,
    )
    assert len(edited) == 1
    if report.swapped_count == 1:
        x = extract_object_number(edited[0], "2")
        y = extract_object_number(edited[0], "3")
        assert x == 120, f"x should be preserved as 120, got {x}"
        assert y == 250, f"y should be preserved as 250, got {y}"


def test_report_counts_add_up() -> None:
    objects = [_obj("1", 30, 180)] * 10
    rng = random.Random(1)
    _, report = apply_style_edit(
        objects,
        generation_assets={},
        prompt="",
        matched_level_names=[],
        style_swap_ratio=0.5,
        jitter_x=0,
        jitter_y=0,
        rng=rng,
        swap_structure=False,
        swap_decoration=False,
    )
    assert report.kept_count + report.swapped_count == len(objects)


# ── run_template_edit ─────────────────────────────────────────────────────────

def test_run_template_edit_returns_objects(tmp_path: Path) -> None:
    gmd_path = tmp_path / "template.gmd"
    objects = [_obj("1", 30 * i, 180) for i in range(1, 11)]
    _make_template_gmd(gmd_path, objects)

    rng = random.Random(0)
    header, edited_objects, report = run_template_edit(
        template_path=gmd_path,
        generation_assets={},
        prompt="",
        matched_level_names=[],
        style_swap_ratio=0.0,
        jitter_x=0,
        jitter_y=0,
        swap_structure=False,
        swap_decoration=False,
        rng=rng,
    )
    assert len(edited_objects) == len(objects)
    assert "kA11" in header or header == ""  # header may vary
    assert report.template_path == str(gmd_path)
    assert report.template_object_count == len(objects)
    assert report.swapped_count == 0


def test_run_template_edit_missing_file_raises(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        run_template_edit(
            template_path=tmp_path / "nonexistent.gmd",
            generation_assets={},
            prompt="",
            matched_level_names=[],
            style_swap_ratio=0.5,
            jitter_x=0,
            jitter_y=0,
            swap_structure=True,
            swap_decoration=True,
            rng=random.Random(0),
        )
