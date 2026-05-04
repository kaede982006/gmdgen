from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from gmdgen.generate.evaluator import combine_model_critic_and_deterministic_scores


@dataclass(slots=True)
class CriticOutput:
    overall_quality_score: float = 0.0
    gd_style_score: float = 0.0
    rhythm_score: float = 0.0
    drop_impact_score: float = 0.0
    playability_risk_score: float = 0.0
    visual_interest_score: float = 0.0
    repetition_score: float = 0.0
    top_weaknesses: list[str] = field(default_factory=list)
    concrete_revision_instructions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_ollama_section_critic_prompt(section_candidate: dict[str, Any], *, section_type: str = "normal") -> list[dict[str, str]]:
    system = (
        "You are a Geometry Dash level quality critic. Evaluate the structured plan only. "
        "Do not output raw save strings. Return compact JSON with scores from 0 to 1, "
        "top_weaknesses, and concrete_revision_instructions."
    )
    user = {
        "task": "critic",
        "section_type": section_type,
        "criteria": [
            "GD style",
            "rhythm alignment",
            "drop impact",
            "visual interest",
            "repetition control",
            "playability risk",
        ],
        "candidate": section_candidate,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, sort_keys=True)},
    ]


def parse_critic_response(raw_response: str | dict[str, Any]) -> CriticOutput:
    if isinstance(raw_response, str):
        payload = json.loads(raw_response)
    else:
        payload = dict(raw_response)
    return CriticOutput(
        overall_quality_score=_score(payload.get("overall_quality_score", payload.get("overall", 0.0))),
        gd_style_score=_score(payload.get("gd_style_score", 0.0)),
        rhythm_score=_score(payload.get("rhythm_score", 0.0)),
        drop_impact_score=_score(payload.get("drop_impact_score", 0.0)),
        playability_risk_score=_score(payload.get("playability_risk_score", 0.0)),
        visual_interest_score=_score(payload.get("visual_interest_score", 0.0)),
        repetition_score=_score(payload.get("repetition_score", 0.0)),
        top_weaknesses=[str(item)[:200] for item in payload.get("top_weaknesses", []) if isinstance(item, (str, int, float))][:8],
        concrete_revision_instructions=[
            str(item)[:240] for item in payload.get("concrete_revision_instructions", []) if isinstance(item, (str, int, float))
        ][:8],
    )


def combine_critic_with_deterministic_score(deterministic_score: float, critic: CriticOutput | None) -> float:
    return combine_model_critic_and_deterministic_scores(
        deterministic_score,
        critic.overall_quality_score if critic is not None else None,
        critic_weight=0.35,
    )


def critic_weaknesses_to_feedback(critic: CriticOutput) -> str:
    parts = []
    if critic.top_weaknesses:
        parts.append("Weaknesses: " + "; ".join(critic.top_weaknesses[:4]))
    if critic.concrete_revision_instructions:
        parts.append("Revise by: " + "; ".join(critic.concrete_revision_instructions[:4]))
    return " ".join(parts)


def _score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0
