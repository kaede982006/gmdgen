from __future__ import annotations

from gmdgen.ai.provider import LevelGenerationAIProvider
from gmdgen.ai.schemas import AILevelPlanRequest, AILevelPlanResponse
from gmdgen.gd.plans import SectionPlan
from gmdgen.gd.time_mapping import SpeedState
from gmdgen.generate.section_pipeline import (
    build_global_level_plan,
    build_section_generation_tasks,
    run_section_generation_pipeline,
    select_best_section_candidate,
    weakest_section_candidate,
)


class SequencedProvider(LevelGenerationAIProvider):
    def __init__(self) -> None:
        self.calls = 0

    def generate_level_plan(self, request: AILevelPlanRequest) -> AILevelPlanResponse:
        self.calls += 1
        section = request.section_plans[0]
        section_id = int(section.get("section_id", 0))
        candidate_id = int(request.output_requirements.get("candidate_id", self.calls))
        count = 2 if candidate_id == 1 else 6
        if section.get("section_type") == "drop":
            count += 3
        objects = [
            {
                "object_id": 1,
                "x": section["start_x"] + 30 + idx * 40,
                "y": 90,
                "role": "ai_structure" if idx % 2 else "beat_orb",
                "group_ids": [],
                "beat_aligned_time": section["start_time"],
                "sync_error": 0.0,
            }
            for idx in range(count)
        ]
        return AILevelPlanResponse(object_plans=objects, trigger_plans=[], provider="ollama", model="fake")


def _sections() -> list[SectionPlan]:
    return [
        SectionPlan(0, 4, 0, 480, "intro", "cube", SpeedState.NORMAL, 0.25, 0.2, 0.1, 0.3),
        SectionPlan(4, 8, 480, 960, "drop", "wave", SpeedState.FAST, 0.9, 0.8, 0.8, 0.7),
    ]


def _request() -> AILevelPlanRequest:
    return AILevelPlanRequest(
        project_goal="test",
        generation_mode="audio_conditioned",
        difficulty="normal",
        safe_mode=True,
        object_budget=80,
        song_offset=0.0,
        start_speed="normal",
        audio_summary={},
        beat_summary={},
        onset_summary={},
        section_plans=[],
        time_x_summary={},
        trigger_schema_summary={},
        playability_rules_summary={},
        output_requirements={"total_sections": 2},
    )


def test_section_generation_creates_candidates_per_section() -> None:
    provider = SequencedProvider()
    stitched, candidates = run_section_generation_pipeline(
        provider=provider,
        base_request=_request(),
        section_plans=_sections(),
        candidates_per_section=2,
        object_budget=80,
        max_group_id=999,
        safe_mode=True,
    )

    assert len(candidates) == 4
    assert provider.calls == 4
    assert stitched.global_object_plans


def test_best_section_candidate_selected_by_score() -> None:
    provider = SequencedProvider()
    _stitched, candidates = run_section_generation_pipeline(
        provider=provider,
        base_request=_request(),
        section_plans=_sections()[:1],
        candidates_per_section=2,
        object_budget=80,
        max_group_id=999,
        safe_mode=True,
    )

    best = select_best_section_candidate(candidates)
    scores = [candidate.score_breakdown["total"] for candidate in candidates]
    assert best.selected is True
    assert best.score_breakdown["total"] == max(scores)


def test_section_stitching_preserves_x_order() -> None:
    provider = SequencedProvider()
    stitched, _candidates = run_section_generation_pipeline(
        provider=provider,
        base_request=_request(),
        section_plans=_sections(),
        candidates_per_section=1,
        object_budget=80,
        max_group_id=999,
        safe_mode=True,
    )

    xs = [plan.x for plan in stitched.global_object_plans]
    assert xs == sorted(xs)
    assert stitched.global_consistency_report.x_order_preserved is True


def test_global_consistency_pass_runs_after_stitching() -> None:
    provider = SequencedProvider()
    stitched, _candidates = run_section_generation_pipeline(
        provider=provider,
        base_request=_request(),
        section_plans=_sections(),
        candidates_per_section=1,
        object_budget=80,
        max_group_id=999,
        safe_mode=True,
    )

    assert stitched.global_consistency_report.object_count == len(stitched.global_object_plans)


def test_drop_section_receives_more_candidates_or_higher_density() -> None:
    tasks = build_section_generation_tasks(_sections())
    global_plan = build_global_level_plan(_sections(), song_bpm=120, style_tags=["glow"])

    assert tasks[1].section_type == "drop"
    assert tasks[1].density_target > tasks[0].density_target
    assert global_plan.drop_positions == [4]


def test_weakest_section_selected_for_retry() -> None:
    provider = SequencedProvider()
    _stitched, candidates = run_section_generation_pipeline(
        provider=provider,
        base_request=_request(),
        section_plans=_sections(),
        candidates_per_section=2,
        object_budget=80,
        max_group_id=999,
        safe_mode=True,
    )

    weakest = weakest_section_candidate(candidates)
    assert weakest is not None
    assert weakest.selected is True
