# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from gmdgen.ai.planner import build_template_level_plan
from gmdgen.data.preprocess import split_level_objects
from gmdgen.generate.allocator import allocate_symbols
from gmdgen.generate.decoder import level_plan_to_ir
from gmdgen.generate.pipeline import ALGORITHMIC_SOURCE_OF_TRUTH, serialize_level_ir


def test_source_of_truth_order_is_explicit() -> None:
    assert ALGORITHMIC_SOURCE_OF_TRUTH[:3] == (
        "UserPrompt",
        "GenerationConfig",
        "Ollama SectionPlan JSON",
    )
    assert "Serializer" in ALGORITHMIC_SOURCE_OF_TRUTH
    assert ALGORITHMIC_SOURCE_OF_TRUTH[-1] == "GenerationReport"


def test_section_plan_decodes_to_local_ir() -> None:
    plan = build_template_level_plan(level_name="ir-test", object_budget=80)
    level_ir = level_plan_to_ir(plan)

    assert level_ir.sections
    section_ir = level_ir.sections[0]
    assert section_ir.objects
    assert section_ir.group_symbols[0].name == "intro_blocks"
    assert section_ir.objects[0].group_ids == []


def test_symbolic_groups_and_colors_are_allocated_locally() -> None:
    plan = build_template_level_plan(level_name="alloc-test", object_budget=80)
    level_ir = level_plan_to_ir(plan)

    report = allocate_symbols(level_ir, first_group_id=10, first_color_channel_id=20)

    assert report.passed is True
    assert report.group_ids["intro_blocks"] == 10
    assert report.color_channel_ids["accent_primary"] == 20
    assert level_ir.sections[0].objects[0].group_ids == [10]
    assert level_ir.sections[0].objects[0].color_channel_id == 20


def test_serializer_uses_ir_and_round_trip_object_count_matches() -> None:
    plan = build_template_level_plan(level_name="serialize-test", object_budget=80)
    level_ir = level_plan_to_ir(plan)

    result = serialize_level_ir(level_ir)

    parsed = split_level_objects(result.decoded_gmd)
    assert parsed
    assert len(parsed) == result.report.final_objects
    assert result.report.serialized_objects == result.report.final_objects
    assert result.report.final_success is True
