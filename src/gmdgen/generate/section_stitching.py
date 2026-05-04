from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from gmdgen.gd.plans import ObjectPlan, TriggerPlan


@dataclass(slots=True)
class GlobalConsistencyReport:
    x_order_preserved: bool = True
    object_count: int = 0
    trigger_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StitchedLevelPlan:
    selected_sections: list[dict[str, Any]] = field(default_factory=list)
    global_object_plans: list[ObjectPlan] = field(default_factory=list)
    global_trigger_plans: list[TriggerPlan] = field(default_factory=list)
    speed_plan: list[dict[str, Any]] = field(default_factory=list)
    stitching_warnings: list[str] = field(default_factory=list)
    global_consistency_report: GlobalConsistencyReport = field(default_factory=GlobalConsistencyReport)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_sections": list(self.selected_sections),
            "global_object_count": len(self.global_object_plans),
            "global_trigger_count": len(self.global_trigger_plans),
            "speed_plan": list(self.speed_plan),
            "stitching_warnings": list(self.stitching_warnings),
            "global_consistency_report": self.global_consistency_report.to_dict(),
        }


def stitch_section_candidates(candidates: list[Any]) -> StitchedLevelPlan:
    selected = sorted(candidates, key=lambda candidate: getattr(getattr(candidate, "section_plan", None), "start_x", 0.0))
    objects: list[ObjectPlan] = []
    triggers: list[TriggerPlan] = []
    warnings: list[str] = []
    selected_summary: list[dict[str, Any]] = []
    last_x = -1e18
    for candidate in selected:
        section = candidate.section_plan
        selected_summary.append(
            {
                "section_id": candidate.section_id,
                "candidate_id": candidate.candidate_id,
                "score": candidate.score_breakdown.get("total", 0.0),
            }
        )
        for plan in sorted(candidate.object_plans, key=lambda item: item.x):
            if plan.x < last_x:
                warnings.append(f"x_order_adjusted: section={candidate.section_id}")
                plan.x = last_x
            last_x = plan.x
            plan.safety_flags.setdefault("section_id", candidate.section_id)
            objects.append(plan)
        triggers.extend(sorted(candidate.trigger_plans, key=lambda item: item.x))
    report = run_global_consistency_pass(objects, triggers)
    report.warnings.extend(warnings)
    return StitchedLevelPlan(
        selected_sections=selected_summary,
        global_object_plans=objects,
        global_trigger_plans=triggers,
        stitching_warnings=warnings,
        global_consistency_report=report,
    )


def run_global_consistency_pass(object_plans: list[ObjectPlan], trigger_plans: list[TriggerPlan]) -> GlobalConsistencyReport:
    xs = [plan.x for plan in object_plans]
    x_order = all(xs[idx] <= xs[idx + 1] for idx in range(len(xs) - 1))
    warnings = []
    if not x_order:
        warnings.append("global_x_order_not_sorted")
    if not object_plans:
        warnings.append("global_plan_has_no_objects")
    return GlobalConsistencyReport(
        x_order_preserved=x_order,
        object_count=len(object_plans),
        trigger_count=len(trigger_plans),
        warnings=warnings,
    )
