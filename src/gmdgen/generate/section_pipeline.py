# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

from gmdgen.ai.provider import LevelGenerationAIProvider
from gmdgen.ai.schemas import (
    AILevelPlanRequest,
    AIPlanConversionResult,
    convert_ai_response_to_plans,
)
from gmdgen.gd.geode_bridge import GeodeBridge, NullGeodeBridge
from gmdgen.gd.plans import ObjectPlan, SectionPlan, TriggerPlan
from gmdgen.generate.evaluator import SectionEvaluatorReport, evaluate_geode_quality, evaluate_section_candidate
from gmdgen.generate.renderer import render_plans_with_style
from gmdgen.generate.section_stitching import StitchedLevelPlan, stitch_section_candidates
from gmdgen.generate.style_bank import MotifBank, MotifContext, build_motif_context_for_section


@dataclass(slots=True)
class GlobalLevelPlan:
    level_name: str
    target_duration: float
    song_bpm: float
    song_offset: float
    start_speed: str
    total_sections: int
    global_difficulty_curve: list[float] = field(default_factory=list)
    global_density_curve: list[float] = field(default_factory=list)
    global_trigger_curve: list[float] = field(default_factory=list)
    global_style_tags: list[str] = field(default_factory=list)
    speed_portal_policy: str = "musical"
    mode_progression: list[str] = field(default_factory=list)
    drop_positions: list[float] = field(default_factory=list)
    buildup_positions: list[float] = field(default_factory=list)
    break_positions: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SectionGenerationTask:
    section_id: int
    section_type: str
    start_time: float
    end_time: float
    start_x: float
    end_x: float
    energy_summary: dict[str, Any] = field(default_factory=dict)
    beat_summary: dict[str, Any] = field(default_factory=dict)
    onset_summary: dict[str, Any] = field(default_factory=dict)
    density_target: float = 0.5
    trigger_intensity_target: float = 0.5
    decoration_target: float = 0.5
    difficulty_target: float = 0.5
    allowed_roles: list[str] = field(default_factory=list)
    allowed_triggers: list[str] = field(default_factory=list)
    motif_context: dict[str, Any] = field(default_factory=dict)
    previous_section_summary: dict[str, Any] = field(default_factory=dict)
    next_section_summary: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SectionCandidate:
    section_id: int
    candidate_id: int
    section_plan: SectionPlan
    object_plans: list[ObjectPlan] = field(default_factory=list)
    trigger_plans: list[TriggerPlan] = field(default_factory=list)
    score_breakdown: dict[str, float] = field(default_factory=dict)
    repair_report: dict[str, Any] = field(default_factory=dict)
    evaluator_report: dict[str, Any] = field(default_factory=dict)
    geode_report: dict[str, Any] = field(default_factory=dict)
    selected: bool = False
    rejection_reason: str = ""
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "candidate_id": self.candidate_id,
            "score_breakdown": dict(self.score_breakdown),
            "repair_report": dict(self.repair_report),
            "evaluator_report": dict(self.evaluator_report),
            "geode_report": dict(self.geode_report),
            "selected": self.selected,
            "rejection_reason": self.rejection_reason,
            "object_count": len(self.object_plans),
            "trigger_count": len(self.trigger_plans),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


def build_global_level_plan(
    section_plans: list[SectionPlan],
    *,
    level_name: str = "generated_level",
    target_duration: float = 0.0,
    song_bpm: float = 0.0,
    song_offset: float = 0.0,
    start_speed: str = "normal",
    speed_portal_policy: str = "musical",
    style_tags: list[str] | None = None,
) -> GlobalLevelPlan:
    return GlobalLevelPlan(
        level_name=level_name,
        target_duration=target_duration or max((section.end_time for section in section_plans), default=0.0),
        song_bpm=song_bpm,
        song_offset=song_offset,
        start_speed=start_speed,
        total_sections=len(section_plans),
        global_difficulty_curve=[section.difficulty_target for section in section_plans],
        global_density_curve=[section.density_target for section in section_plans],
        global_trigger_curve=[section.trigger_intensity for section in section_plans],
        global_style_tags=list(style_tags or []),
        speed_portal_policy=speed_portal_policy,
        mode_progression=[section.gameplay_mode for section in section_plans],
        drop_positions=[section.start_time for section in section_plans if section.section_type == "drop"],
        buildup_positions=[section.start_time for section in section_plans if section.section_type == "buildup"],
        break_positions=[section.start_time for section in section_plans if section.section_type == "break"],
    )


def build_section_generation_tasks(
    section_plans: list[SectionPlan],
    *,
    motif_bank: MotifBank | None = None,
    allowed_roles: list[str] | None = None,
    allowed_triggers: list[str] | None = None,
) -> list[SectionGenerationTask]:
    tasks: list[SectionGenerationTask] = []
    for idx, section in enumerate(section_plans):
        tasks.append(
            SectionGenerationTask(
                section_id=idx,
                section_type=section.section_type,
                start_time=section.start_time,
                end_time=section.end_time,
                start_x=section.start_x,
                end_x=section.end_x,
                energy_summary={"density_target": section.density_target},
                density_target=section.density_target,
                trigger_intensity_target=section.trigger_intensity,
                decoration_target=section.decoration_intensity,
                difficulty_target=section.difficulty_target,
                allowed_roles=list(allowed_roles or []),
                allowed_triggers=list(allowed_triggers or []),
                motif_context=build_motif_context_for_section(motif_bank, idx, section.section_type).to_dict(),
                previous_section_summary=_section_summary(section_plans[idx - 1]) if idx > 0 else {},
                next_section_summary=_section_summary(section_plans[idx + 1]) if idx + 1 < len(section_plans) else {},
                constraints={
                    "x_bounds": [section.start_x, section.end_x],
                    "do_not_leave_drop_empty": section.section_type == "drop",
                    "prefer_density_target": section.density_target,
                },
            )
        )
    return tasks


def generate_section_candidates(
    *,
    provider: LevelGenerationAIProvider,
    base_request: AILevelPlanRequest,
    task: SectionGenerationTask,
    section_plan: SectionPlan,
    candidates_per_section: int,
    object_budget: int,
    max_group_id: int,
    safe_mode: bool,
    style_summary: dict[str, Any] | None = None,
    geode_bridge: GeodeBridge | None = None,
) -> list[SectionCandidate]:
    candidates: list[SectionCandidate] = []
    per_section_budget = max(1, object_budget // max(1, int(base_request.output_requirements.get("total_sections", 1))))
    for candidate_id in range(1, max(1, candidates_per_section) + 1):
        request = _request_for_section(base_request, task, per_section_budget, candidate_id)
        response = provider.generate_level_plan(request)
        conversion = convert_ai_response_to_plans(
            response,
            object_budget=per_section_budget,
            max_group_id=max_group_id,
            safe_mode=safe_mode,
            section_plans=[section_plan],
        )
        objects = _clip_objects_to_section(conversion.object_plans, section_plan)
        triggers = _clip_triggers_to_section(conversion.trigger_plans, section_plan)
        render_plans_with_style(objects, [section_plan], style_summary=style_summary or {}, safe_mode=safe_mode, seed=candidate_id)
        geode_report = evaluate_geode_quality(geode_bridge or NullGeodeBridge())
        evaluator_report: SectionEvaluatorReport = evaluate_section_candidate(
            section=section_plan,
            object_plans=objects,
            trigger_plans=triggers,
            style_summary=style_summary or {},
            repair_loss_ratio=_repair_loss_ratio(response.object_plans, response.trigger_plans, objects, triggers),
            geode_report=geode_report,
        )
        candidate = SectionCandidate(
            section_id=task.section_id,
            candidate_id=candidate_id,
            section_plan=section_plan,
            object_plans=objects,
            trigger_plans=triggers,
            score_breakdown=evaluator_report.to_dict(),
            evaluator_report=evaluator_report.to_dict(),
            geode_report=geode_report.to_dict(),
            rejection_reason=_reject_reason(evaluator_report, conversion),
            warnings=list(conversion.warnings) + list(evaluator_report.warnings),
            errors=list(conversion.errors),
        )
        candidates.append(candidate)
    return candidates


def select_best_section_candidate(
    candidates: list[SectionCandidate],
    *,
    min_section_score: float = 0.0,
    min_drop_section_score: float = 0.0,
) -> SectionCandidate:
    if not candidates:
        raise ValueError("section candidate list is empty")
    selected = max(candidates, key=lambda candidate: float(candidate.score_breakdown.get("total", 0.0)))
    threshold = min_drop_section_score if selected.section_plan.section_type == "drop" else min_section_score
    if float(selected.score_breakdown.get("total", 0.0)) < threshold:
        selected.rejection_reason = selected.rejection_reason or "below_section_quality_threshold"
    for candidate in candidates:
        candidate.selected = candidate is selected
    return selected


def run_section_generation_pipeline(
    *,
    provider: LevelGenerationAIProvider,
    base_request: AILevelPlanRequest,
    section_plans: list[SectionPlan],
    motif_bank: MotifBank | None = None,
    candidates_per_section: int = 4,
    object_budget: int = 1200,
    max_group_id: int = 9999,
    safe_mode: bool = True,
    style_summary: dict[str, Any] | None = None,
    geode_bridge: GeodeBridge | None = None,
    min_section_score: float = 0.35,
    min_drop_section_score: float = 0.45,
) -> tuple[StitchedLevelPlan, list[SectionCandidate]]:
    tasks = build_section_generation_tasks(section_plans, motif_bank=motif_bank)
    all_candidates: list[SectionCandidate] = []
    selected: list[SectionCandidate] = []
    for task, section in zip(tasks, section_plans):
        candidates = generate_section_candidates(
            provider=provider,
            base_request=base_request,
            task=task,
            section_plan=section,
            candidates_per_section=candidates_per_section,
            object_budget=object_budget,
            max_group_id=max_group_id,
            safe_mode=safe_mode,
            style_summary=style_summary or {},
            geode_bridge=geode_bridge or NullGeodeBridge(),
        )
        all_candidates.extend(candidates)
        selected.append(select_best_section_candidate(candidates, min_section_score=min_section_score, min_drop_section_score=min_drop_section_score))
    return stitch_section_candidates(selected), all_candidates


def weakest_section_candidate(candidates: list[SectionCandidate]) -> SectionCandidate | None:
    selected = [candidate for candidate in candidates if candidate.selected]
    if not selected:
        return None
    return min(selected, key=lambda candidate: float(candidate.score_breakdown.get("total", 0.0)))


def _request_for_section(base_request: AILevelPlanRequest, task: SectionGenerationTask, object_budget: int, candidate_id: int) -> AILevelPlanRequest:
    output_requirements = dict(base_request.output_requirements)
    output_requirements.update(
        {
            "section_generation_task": task.to_dict(),
            "candidate_id": candidate_id,
            "object_budget": object_budget,
            "generate_one_section_only": True,
        }
    )
    return replace(
        base_request,
        project_goal=f"{base_request.project_goal} Generate section {task.section_id} ({task.section_type}) only.",
        object_budget=object_budget,
        section_plans=[task.to_dict()],
        style_reference_summary={**base_request.style_reference_summary, "motif_context": task.motif_context},
        output_requirements=output_requirements,
    )


def _clip_objects_to_section(object_plans: list[ObjectPlan], section: SectionPlan) -> list[ObjectPlan]:
    result = []
    for plan in object_plans:
        if section.start_x - 30 <= plan.x <= section.end_x + 30:
            plan.x = min(section.end_x, max(section.start_x, plan.x))
            result.append(plan)
    return result


def _clip_triggers_to_section(trigger_plans: list[TriggerPlan], section: SectionPlan) -> list[TriggerPlan]:
    result = []
    for plan in trigger_plans:
        if section.start_x - 60 <= plan.x <= section.end_x + 60:
            plan.x = min(section.end_x, max(section.start_x, plan.x))
            result.append(plan)
    return result


def _repair_loss_ratio(raw_objects: list[dict[str, Any]], raw_triggers: list[dict[str, Any]], objects: list[ObjectPlan], triggers: list[TriggerPlan]) -> float:
    raw_count = max(1, len(raw_objects) + len(raw_triggers))
    final_count = len(objects) + len(triggers)
    return max(0.0, min(1.0, (raw_count - final_count) / raw_count))


def _reject_reason(evaluator_report: SectionEvaluatorReport, conversion: AIPlanConversionResult) -> str:
    if conversion.errors:
        return "; ".join(conversion.errors[:3])
    if evaluator_report.warnings:
        return "; ".join(evaluator_report.warnings[:3])
    if evaluator_report.total < 0.35:
        return "low_section_score"
    return ""


def _section_summary(section: SectionPlan) -> dict[str, Any]:
    return {
        "section_type": section.section_type,
        "start_time": section.start_time,
        "end_time": section.end_time,
        "density_target": section.density_target,
        "trigger_intensity": section.trigger_intensity,
    }
