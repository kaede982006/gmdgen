# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class GroupSymbol:
    name: str


@dataclass(frozen=True, slots=True)
class ColorSymbol:
    name: str


@dataclass(slots=True)
class GenerationConfig:
    prompt: str = ""
    level_name: str = "generated_level"
    difficulty: str = "normal"
    target_duration: float = 30.0
    object_budget: int = 500
    style: str = "modern_glow"
    sync_intensity: str = "medium"
    seed: int = 42


@dataclass(slots=True)
class SectionPlan:
    section_id: str
    time_start: float
    time_end: float
    game_mode: str
    speed: str
    density: float
    primary_pattern: str
    allowed_object_families: list[str] = field(default_factory=list)
    forbidden_features: list[str] = field(default_factory=list)
    trigger_budget: int = 0
    group_symbols: list[GroupSymbol] = field(default_factory=list)
    color_symbols: list[ColorSymbol] = field(default_factory=list)
    design_notes: str = ""


@dataclass(slots=True)
class LevelPlan:
    level_name: str
    difficulty: str
    target_duration: float
    object_budget: int
    style: str
    sync_intensity: str
    sections: list[SectionPlan] = field(default_factory=list)


@dataclass(slots=True)
class GMDObjectIR:
    object_id: str
    x: float
    y: float
    role: str
    group_symbols: list[GroupSymbol] = field(default_factory=list)
    color_symbol: ColorSymbol | None = None
    group_ids: list[int] = field(default_factory=list)
    color_channel_id: int | None = None
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TriggerIR:
    trigger_type: str
    x: float
    y: float
    target_group_symbol: GroupSymbol | None = None
    color_symbol: ColorSymbol | None = None
    target_group_id: int | None = None
    color_channel_id: int | None = None
    duration: float = 0.0
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SectionIR:
    section_id: str
    time_start: float
    time_end: float
    game_mode: str
    speed: str
    density: float
    objects: list[GMDObjectIR] = field(default_factory=list)
    triggers: list[TriggerIR] = field(default_factory=list)
    group_symbols: list[GroupSymbol] = field(default_factory=list)
    color_symbols: list[ColorSymbol] = field(default_factory=list)
    source_plan: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LevelIR:
    level_name: str
    difficulty: str
    target_duration: float
    object_budget: int
    style: str
    sync_intensity: str
    sections: list[SectionIR] = field(default_factory=list)

    @property
    def object_count(self) -> int:
        return sum(len(section.objects) for section in self.sections)

    @property
    def trigger_count(self) -> int:
        return sum(len(section.triggers) for section in self.sections)


@dataclass(slots=True)
class ValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "metrics": dict(self.metrics),
        }


@dataclass(slots=True)
class RepairResult:
    applied: bool = False
    repair_loss: float = 0.0
    metrics_updated: bool = True
    changes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GenerationArtifact:
    config: GenerationConfig
    level_plan: LevelPlan | None = None
    level_ir: LevelIR | None = None
    serialized_gmd: str = ""
    report: "GenerationReport | None" = None


@dataclass(slots=True)
class GenerationReport:
    planner_status: str = "not_used"
    planner_fallback_used: bool = False
    candidate_ir_objects: int = 0
    serialized_objects: int = 0
    final_objects: int = 0
    syntax_validation: ValidationResult = field(default_factory=lambda: ValidationResult(False))
    semantic_validation: ValidationResult = field(default_factory=lambda: ValidationResult(False))
    playability_validation: ValidationResult = field(default_factory=lambda: ValidationResult(False))
    repair_applied: bool = False
    repair_loss: float = 0.0
    quality_gate_passed: bool = False
    low_quality_draft_saved: bool = False
    final_success: bool = False
    consistency_errors: list[str] = field(default_factory=list)

    # Compute & GPU reporting
    compute_device: str = "cpu"
    gpu_available: bool = False
    gpu_name: str | None = None
    gpu_backend: str | None = None
    gpu_used_for_training: bool = False
    gpu_used_for_generation: bool = False
    gpu_used_for_embeddings: bool = False
    gpu_fallback_reason: str | None = None
    torch_available: bool = False
    cuda_available: bool = False
    mps_available: bool = False
    ollama_model: str | None = None
    ollama_gpu_status_known: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner_status": self.planner_status,
            "planner_fallback_used": self.planner_fallback_used,
            "candidate_ir_objects": self.candidate_ir_objects,
            "serialized_objects": self.serialized_objects,
            "final_objects": self.final_objects,
            "syntax_validation": self.syntax_validation.to_dict(),
            "semantic_validation": self.semantic_validation.to_dict(),
            "playability_validation": self.playability_validation.to_dict(),
            "repair_applied": self.repair_applied,
            "repair_loss": self.repair_loss,
            "quality_gate_passed": self.quality_gate_passed,
            "low_quality_draft_saved": self.low_quality_draft_saved,
            "final_success": self.final_success,
            "consistency_errors": list(self.consistency_errors),
            # GPU status
            "compute_device": self.compute_device,
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name,
            "gpu_backend": self.gpu_backend,
            "gpu_used_for_training": self.gpu_used_for_training,
            "gpu_used_for_generation": self.gpu_used_for_generation,
            "gpu_used_for_embeddings": self.gpu_used_for_embeddings,
            "gpu_fallback_reason": self.gpu_fallback_reason,
            "torch_available": self.torch_available,
            "cuda_available": self.cuda_available,
            "mps_available": self.mps_available,
            "ollama_model": self.ollama_model,
            "ollama_gpu_status_known": self.ollama_gpu_status_known,
        }
