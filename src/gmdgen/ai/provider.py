from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from gmdgen.ai.schemas import AILevelPlanRequest, AILevelPlanResponse


@dataclass(slots=True)
class AIProviderUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class AIProviderResult:
    provider_name: str
    model_name: str
    raw_response_summary: str = ""
    parsed_plan: AILevelPlanResponse | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    latency_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    cache_hit: bool = False
    request_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "raw_response_summary": self.raw_response_summary,
            "parsed_plan": self.parsed_plan.to_dict() if self.parsed_plan else None,
            "usage": dict(self.usage),
            "latency_seconds": self.latency_seconds,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "cache_hit": self.cache_hit,
            "request_hash": self.request_hash,
        }


class LevelGenerationAIProvider(ABC):
    provider_name: str = "unknown"
    model_name: str = ""

    def supports_structured_output(self) -> bool:
        return True

    def supports_json_schema(self) -> bool:
        return True

    def classify_error(self, error: BaseException) -> str:
        return "unknown_error"

    def sanitize_config_for_log(self) -> dict[str, Any]:
        return {"provider_name": self.provider_name, "model_name": self.model_name}

    def estimate_request_tokens(self, request: AILevelPlanRequest) -> int:
        return max(1, len(str(request.to_dict())) // 4)

    def build_request_payload(self, request: AILevelPlanRequest) -> dict[str, Any]:
        return request.to_dict()

    def parse_response(self, response: Any) -> AILevelPlanResponse:
        from gmdgen.ai.schemas import parse_ai_level_plan_response

        return parse_ai_level_plan_response(response)

    @abstractmethod
    def generate_level_plan(self, request: AILevelPlanRequest) -> AILevelPlanResponse:
        raise NotImplementedError


class LocalHeuristicProvider(LevelGenerationAIProvider):
    """Fallback provider that keeps the existing rule-based planner in control."""

    provider_name = "local_test_only"
    model_name = "local-heuristic"

    def __init__(self, *, reason: str = "local heuristic planner") -> None:
        self.reason = reason

    def generate_level_plan(self, request: AILevelPlanRequest) -> AILevelPlanResponse:
        return AILevelPlanResponse(
            sections=request.section_plans,
            object_plans=[],
            trigger_plans=[],
            reasoning_summary="Using local audio-conditioned heuristic planner.",
            safety_notes=["No remote model output was used."],
            metadata={"local_reason": self.reason},
            provider="local",
            model="local-heuristic",
        )
