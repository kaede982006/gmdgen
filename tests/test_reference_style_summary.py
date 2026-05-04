from __future__ import annotations

from gmdgen.ai.context import summarize_reference_level
from gmdgen.ai.prompts import summarize_reference_style_for_model


def test_reference_summary_includes_density_and_ratios() -> None:
    level = "1,1,2,0,3,90;1,500,2,60,3,240;1,36,2,120,3,180;1,1006,2,120,3,300,51,1;"

    summary = summarize_reference_level(level)

    assert "structure_object_ratio" in summary
    assert "decoration_object_ratio" in summary
    assert "gameplay_object_ratio" in summary
    assert "average_density" in summary


def test_reference_summary_includes_trigger_distribution() -> None:
    summary = summarize_reference_level("1,1006,2,120,3,300,51,1;1,901,2,180,3,300,51,1;")

    assert summary["trigger_count"] == 2
    assert summary["trigger_type_distribution"]["1006"] == 1


def test_reference_summary_does_not_dump_full_level_string() -> None:
    raw = ";".join(f"1,1,2,{idx},3,90" for idx in range(100))
    summary = summarize_reference_level(raw)

    assert summary["text_included"] is False
    assert raw not in str(summary)


def test_motif_summary_extracted_from_reference_level() -> None:
    raw = ";".join(f"1,{1 if idx % 2 else 500},2,{idx*30},3,90" for idx in range(24))
    summary = summarize_reference_level(raw)

    assert summary["common_motif_patterns"]
    assert "motif_id" in summary["common_motif_patterns"][0]


def test_reference_prompt_summary_includes_density_and_ratios() -> None:
    style = {
        "object_count": 10,
        "trigger_count": 2,
        "structure_object_ratio": 0.5,
        "decoration_object_ratio": 0.3,
        "gameplay_object_ratio": 0.2,
        "trigger_ratio": 0.2,
        "object_id_distribution": {"1": 5},
        "role_distribution": {"structure": 5},
        "common_motif_patterns": [{"motif_id": "m0"}],
    }

    prompt_summary = summarize_reference_style_for_model(style)

    assert prompt_summary["structure_object_ratio"] == 0.5
    assert prompt_summary["common_motif_patterns"][0]["motif_id"] == "m0"
