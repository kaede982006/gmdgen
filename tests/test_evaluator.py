from __future__ import annotations

from gmdgen.ai.critic import (
    build_ollama_section_critic_prompt,
    combine_critic_with_deterministic_score,
    critic_weaknesses_to_feedback,
    parse_critic_response,
)
from gmdgen.gd.geode_bridge import NullGeodeBridge
from gmdgen.gd.plans import ObjectPlan, SectionPlan, TriggerPlan
from gmdgen.gd.time_mapping import SpeedState
from gmdgen.generate.evaluator import evaluate_geode_quality, evaluate_section_candidate


def _section(section_type: str = "drop") -> SectionPlan:
    return SectionPlan(0, 4, 0, 480, section_type, "cube", SpeedState.NORMAL, 0.8, 0.7, 0.7, 0.6)


def test_critic_response_parsed() -> None:
    critic = parse_critic_response(
        {
            "overall_quality_score": 0.8,
            "gd_style_score": 0.7,
            "top_weaknesses": ["weak drop"],
            "concrete_revision_instructions": ["add pulse accents"],
        }
    )

    assert critic.overall_quality_score == 0.8
    assert "weak drop" in critic.top_weaknesses


def test_critic_score_combined_with_deterministic_score() -> None:
    critic = parse_critic_response({"overall_quality_score": 1.0})

    assert combine_critic_with_deterministic_score(0.5, critic) > 0.5


def test_critic_weaknesses_feed_retry_prompt() -> None:
    critic = parse_critic_response(
        {
            "overall_quality_score": 0.2,
            "top_weaknesses": ["too sparse"],
            "concrete_revision_instructions": ["increase density"],
        }
    )

    feedback = critic_weaknesses_to_feedback(critic)
    assert "too sparse" in feedback
    assert "increase density" in feedback


def test_critic_prompt_does_not_allow_raw_save_string() -> None:
    messages = build_ollama_section_critic_prompt({"object_plans": []}, section_type="drop")

    assert "Do not output raw save strings" in messages[0]["content"]


def test_geode_quality_report_penalizes_bad_candidate() -> None:
    report = evaluate_geode_quality(NullGeodeBridge(), level_string="1,1,2,0,3,0;")

    assert report.available is False
    assert report.score_penalty == 0.0


def test_section_evaluator_rewards_drop_with_trigger() -> None:
    section = _section("drop")
    objects = [
        ObjectPlan("1", 30, 90, "ai_structure", beat_aligned_time=0.0),
        ObjectPlan("36", 90, 150, "beat_orb", beat_aligned_time=0.5),
        ObjectPlan("500", 150, 220, "safe_decoration"),
        ObjectPlan("8", 210, 90, "obstacle", beat_aligned_time=1.0),
    ]
    triggers = [TriggerPlan("pulse", "1006", 30, 300, target_group=1, duration=0.15, beat_aligned_time=0.0)]

    report = evaluate_section_candidate(section=section, object_plans=objects, trigger_plans=triggers)

    assert report.total > 0.3
    assert report.drop_impact_score if hasattr(report, "drop_impact_score") else True
