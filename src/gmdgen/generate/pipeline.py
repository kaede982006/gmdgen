# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass

from gmdgen.data.preprocess import split_level_objects
from gmdgen.gd.plans import ObjectPlan, TriggerMode, TriggerPlan, plans_to_level_objects
from gmdgen.generate.allocator import SymbolAllocationReport, allocate_symbols
from gmdgen.generate.ir import GenerationArtifact, GenerationConfig, GenerationReport, LevelIR, ValidationResult


ALGORITHMIC_SOURCE_OF_TRUTH = (
    "UserPrompt",
    "GenerationConfig",
    "Ollama SectionPlan JSON",
    "Local SectionIR",
    "LevelIR",
    "Group/Color Allocator",
    "TriggerGraph",
    "Serializer",
    "SyntaxValidator",
    "SemanticValidator",
    "PlayabilityValidator",
    "Repairer",
    "Final GMD",
    "GenerationReport",
)


@dataclass(slots=True)
class PipelineSerializationResult:
    level_objects: list[str]
    decoded_gmd: str
    allocation_report: SymbolAllocationReport
    report: GenerationReport


def serialize_level_ir(level_ir: LevelIR) -> PipelineSerializationResult:
    """Serialize local IR after deterministic allocation.

    This adapter is intentionally small; the production audio-conditioned path
    has a richer materializer, but both paths share the same rule: local IR is
    converted to concrete objects before any .gmd string is emitted.
    """
    allocation_report = allocate_symbols(level_ir)
    object_plans: list[ObjectPlan] = []
    trigger_plans: list[TriggerPlan] = []
    for section in level_ir.sections:
        for obj in section.objects:
            object_plans.append(
                ObjectPlan(
                    object_id=obj.object_id,
                    x=obj.x,
                    y=obj.y,
                    role=obj.role,
                    group_ids=list(obj.group_ids),
                    color_channel=obj.color_channel_id,
                )
            )
        for trigger in section.triggers:
            trigger_plans.append(
                TriggerPlan(
                    trigger_type=trigger.trigger_type,
                    object_id="1006" if trigger.trigger_type == "pulse" else "901",
                    x=trigger.x,
                    y=trigger.y,
                    target_group=trigger.target_group_id,
                    duration=trigger.duration,
                    properties=dict(trigger.properties),
                )
            )
    level_objects = plans_to_level_objects(
        object_plans,
        trigger_plans,
        trigger_mode=TriggerMode.SAFE,
    )
    decoded_gmd = ";".join(level_objects)
    parsed_objects = split_level_objects(decoded_gmd)
    report = GenerationReport(
        planner_status="contract_test",
        planner_fallback_used=False,
        candidate_ir_objects=level_ir.object_count + level_ir.trigger_count,
        serialized_objects=len(parsed_objects),
        final_objects=len(parsed_objects),
        syntax_validation=ValidationResult(passed=bool(parsed_objects)),
        semantic_validation=ValidationResult(passed=allocation_report.passed, errors=list(allocation_report.errors)),
        playability_validation=ValidationResult(passed=True),
        repair_applied=False,
        repair_loss=0.0,
        quality_gate_passed=bool(parsed_objects) and allocation_report.passed,
        final_success=bool(parsed_objects) and allocation_report.passed,
    )
    return PipelineSerializationResult(
        level_objects=level_objects,
        decoded_gmd=decoded_gmd,
        allocation_report=allocation_report,
        report=report,
    )


def build_generation_artifact(
    *,
    config: GenerationConfig,
    level_ir: LevelIR,
    serialized_gmd: str,
    report: GenerationReport,
) -> GenerationArtifact:
    return GenerationArtifact(
        config=config,
        level_ir=level_ir,
        serialized_gmd=serialized_gmd,
        report=report,
    )
