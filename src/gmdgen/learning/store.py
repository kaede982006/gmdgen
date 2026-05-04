from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_LEARNING_DIR = Path.home() / ".gmdgen" / "learning"


import enum

class LearningExampleStatus(str, enum.Enum):
    REJECTED = "rejected"
    LOW_QUALITY = "low_quality"
    UNREVIEWED = "unreviewed"
    ACCEPTED = "accepted"
    HIGH_QUALITY = "high_quality"
    USER_FAVORITE = "user_favorite"

@dataclass(slots=True)
class LearningExample:
    example_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    audio_summary: dict[str, Any] = field(default_factory=dict)
    section_summary: list[dict[str, Any]] = field(default_factory=list)
    reference_style_summary: dict[str, Any] = field(default_factory=dict)
    generation_config_summary: dict[str, Any] = field(default_factory=dict)
    prompt_summary: str = ""
    raw_ai_plan_summary: dict[str, Any] = field(default_factory=dict)
    normalized_plan_summary: dict[str, Any] = field(default_factory=dict)
    repaired_plan_summary: dict[str, Any] = field(default_factory=dict)
    final_plan_summary: dict[str, Any] = field(default_factory=dict)
    score_breakdown: dict[str, Any] = field(default_factory=dict)
    candidate_reports: list[dict[str, Any]] = field(default_factory=list)
    selected_candidate_id: int = 0
    quality_loss_reasons: list[str] = field(default_factory=list)
    output_file_name: str = ""
    user_rating: int = 0
    user_tags: list[str] = field(default_factory=list)
    user_notes: str = ""
    accepted_for_training: bool = True
    status: str = LearningExampleStatus.UNREVIEWED.value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LearningMemoryContext:
    good_examples_summary: list[dict[str, Any]] = field(default_factory=list)
    failure_patterns_summary: list[str] = field(default_factory=list)
    preferred_object_ratios: dict[str, float] = field(default_factory=dict)
    common_quality_issues: list[str] = field(default_factory=list)
    successful_prompt_hints: list[str] = field(default_factory=list)
    rejected_patterns: list[str] = field(default_factory=list)
    user_preference_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def learning_store_path(store_dir: str | Path | None = None) -> Path:
    directory = Path(store_dir) if store_dir else DEFAULT_LEARNING_DIR
    return directory.expanduser() / "examples.jsonl"


def save_learning_example(example: LearningExample | dict[str, Any], *, store_dir: str | Path | None = None) -> str:
    payload = sanitize_learning_payload(example.to_dict() if isinstance(example, LearningExample) else dict(example))
    example_id = str(payload.get("example_id") or uuid.uuid4().hex)
    payload["example_id"] = example_id
    path = learning_store_path(store_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return example_id


def update_learning_example_feedback(example_id: str, feedback: dict[str, Any], *, store_dir: str | Path | None = None) -> bool:
    path = learning_store_path(store_dir)
    records = load_learning_examples(store_dir=store_dir, include_corrupt=False)
    updated = False
    sanitized_feedback = sanitize_learning_payload(feedback)
    for record in records:
        if str(record.get("example_id")) == str(example_id):
            record.update(
                {
                    "user_rating": int(sanitized_feedback.get("user_rating", record.get("user_rating", 0)) or 0),
                    "user_tags": list(sanitized_feedback.get("user_tags", record.get("user_tags", [])) or []),
                    "user_notes": str(sanitized_feedback.get("user_notes", record.get("user_notes", "")) or ""),
                    "accepted_for_training": bool(sanitized_feedback.get("accepted_for_training", record.get("accepted_for_training", True))),
                }
            )
            updated = True
            break
    if updated:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(sanitize_learning_payload(record), ensure_ascii=False, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
    return updated


def load_learning_examples(
    *,
    store_dir: str | Path | None = None,
    limit: int | None = None,
    filters: dict[str, Any] | None = None,
    include_corrupt: bool = False,
) -> list[dict[str, Any]]:
    path = learning_store_path(store_dir)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            if include_corrupt:
                records.append({"corrupt": True})
            continue
        if _matches_filters(payload, filters or {}):
            records.append(payload)
    if limit is not None:
        return records[-max(0, int(limit)) :]
    return records


def select_high_quality_examples(examples: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    eligible = []
    for item in examples:
        status = item.get("status", "unreviewed")
        if status in {"rejected", "low_quality"}:
            continue
        if not item.get("accepted_for_training", True):
            continue
        rating = int(item.get("user_rating", 0) or 0)
        score = float((item.get("score_breakdown", {}) or {}).get("total", 0.0) or 0.0)
        
        # New strict threshold
        if rating >= 4 or (score >= 0.85 and status in {"high_quality", "accepted"}):
            eligible.append(item)
            
    eligible.sort(key=lambda item: (int(item.get("user_rating", 0) or 0), float((item.get("score_breakdown", {}) or {}).get("total", 0.0) or 0.0)), reverse=True)
    return eligible[:limit]


def quarantine_bad_learning_examples(*, store_dir: str | Path | None = None) -> int:
    path = learning_store_path(store_dir)
    if not path.exists():
        return 0
    records = []
    quarantined = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            status = payload.get("status", "unreviewed")
            # If description is garbage, repair ratio is too high, or explicitly rejected
            repair_report = payload.get("repaired_plan_summary", {})
            repair_loss = repair_report.get("repair_loss_ratio", 0.0) if isinstance(repair_report, dict) else 0.0
            
            if status == "rejected" or repair_loss > 0.6:
                payload["status"] = LearningExampleStatus.REJECTED.value
                quarantined += 1
            elif status == "unreviewed" and repair_loss > 0.4:
                payload["status"] = LearningExampleStatus.LOW_QUALITY.value
                quarantined += 1
            
            records.append(json.dumps(payload, ensure_ascii=False))
        except Exception:
            quarantined += 1
            continue
    if quarantined > 0:
        path.write_text("\n".join(records) + "\n", encoding="utf-8")
    return quarantined

def select_failure_examples_for_prompt_feedback(examples: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    failures = []
    for item in examples:
        status = item.get("status", "unreviewed")
        rating = int(item.get("user_rating", 0) or 0)
        if rating in {1, 2} or status in {"rejected", "low_quality"} or item.get("quality_loss_reasons") or "quality_gate_failed" in str(item.get("final_plan_summary", {})):
            failures.append(item)
    return failures[-limit:]


def summarize_learning_examples_for_context(
    *,
    store_dir: str | Path | None = None,
    max_chars: int = 2500,
) -> LearningMemoryContext:
    examples = load_learning_examples(store_dir=store_dir, limit=80)
    good = select_high_quality_examples(examples, limit=5)
    failures = select_failure_examples_for_prompt_feedback(examples, limit=5)
    context = LearningMemoryContext(
        good_examples_summary=[_compact_good_example(item) for item in good],
        failure_patterns_summary=[_failure_summary(item) for item in failures],
        common_quality_issues=_top_quality_issues(examples),
        successful_prompt_hints=[str(item.get("prompt_summary", ""))[:160] for item in good if item.get("prompt_summary")],
        rejected_patterns=[tag for item in failures for tag in item.get("user_tags", [])][:12],
        user_preference_summary=_preference_summary(examples),
    )
    payload = context.to_dict()
    text = json.dumps(payload, ensure_ascii=False)
    if len(text) <= max_chars:
        return context
    context.good_examples_summary = context.good_examples_summary[:2]
    context.failure_patterns_summary = context.failure_patterns_summary[:3]
    context.successful_prompt_hints = context.successful_prompt_hints[:3]
    return context


def build_learning_example_from_result(result: dict[str, Any], config_summary: dict[str, Any]) -> LearningExample:
    validation = result.get("validation_report", {}) if isinstance(result.get("validation_report", {}), dict) else {}
    snapshots = validation.get("plan_snapshots", []) if isinstance(validation.get("plan_snapshots", []), list) else []
    def stage(name: str) -> dict[str, Any]:
        for snapshot in snapshots:
            if isinstance(snapshot, dict) and snapshot.get("stage") == name:
                return _small_snapshot(snapshot)
        return {}

    return LearningExample(
        audio_summary={
            "audio_file_name": result.get("audio_file_name", ""),
            "audio_backend": result.get("audio_backend", ""),
            "detected_bpm": result.get("detected_bpm", result.get("bpm", 0.0)),
            "beat_count": result.get("beat_count", result.get("num_beats", 0)),
            "onset_count": result.get("onset_count", result.get("num_onsets", 0)),
            "section_count": result.get("section_count", result.get("num_sections", 0)),
        },
        section_summary=list(result.get("section_plan", []))[:16],
        reference_style_summary=dict(result.get("style_reference_summary", {})) if isinstance(result.get("style_reference_summary", {}), dict) else {},
        generation_config_summary=sanitize_learning_payload(config_summary),
        prompt_summary=str(config_summary.get("prompt", ""))[:500],
        raw_ai_plan_summary=stage("raw_ai_plan"),
        normalized_plan_summary=stage("normalized_plan"),
        repaired_plan_summary=stage("repaired_plan"),
        final_plan_summary=stage("final_encoded_plan"),
        score_breakdown=dict(result.get("score_breakdown", result.get("score", {}))) if isinstance(result.get("score_breakdown", result.get("score", {})), dict) else {},
        candidate_reports=list(result.get("candidate_reports", []))[:12],
        selected_candidate_id=int(result.get("selected_candidate_id", 0) or 0),
        quality_loss_reasons=list(result.get("quality_loss_reason_summary", []))[:12],
        output_file_name=Path(str(result.get("output_path", ""))).name,
        accepted_for_training=result.get("valid", True),
        status="unreviewed" if result.get("valid", True) else "rejected",
    )


def sanitize_learning_payload(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            if "api_key" in lowered or "openai_key" in lowered or "base_url" in lowered or "host" in lowered:
                continue
            sanitized[key_text] = sanitize_learning_payload(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_learning_payload(item) for item in value[:200]]
    if isinstance(value, str):
        if value.startswith("sk-"):
            return "sk-[REDACTED]"
        normalized = value.replace("\\", "/")
        if ":" in normalized[:4] and "/" in normalized:
            return Path(normalized).name
        if len(value) > 4000:
            return value[:4000] + "...[truncated]"
    return value


def _matches_filters(payload: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        if payload.get(key) != expected:
            return False
    return True


def _small_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": snapshot.get("stage", ""),
        "object_count": snapshot.get("object_count", 0),
        "trigger_count": snapshot.get("trigger_count", 0),
        "role_distribution": snapshot.get("role_distribution", {}),
        "trigger_type_distribution": snapshot.get("trigger_type_distribution", {}),
    }


def _compact_good_example(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "rating": item.get("user_rating", 0),
        "score": (item.get("score_breakdown", {}) or {}).get("total", 0.0),
        "prompt_hint": str(item.get("prompt_summary", ""))[:160],
        "final_plan_summary": item.get("final_plan_summary", {}),
        "quality_tags": item.get("user_tags", [])[:8],
    }


def _failure_summary(item: dict[str, Any]) -> str:
    tags = ", ".join(item.get("user_tags", [])[:6])
    reasons = "; ".join(item.get("quality_loss_reasons", [])[:3])
    notes = str(item.get("user_notes", ""))[:160]
    return f"rating={item.get('user_rating', 0)} tags=[{tags}] reasons={reasons} notes={notes}".strip()


def _top_quality_issues(examples: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for item in examples:
        for tag in item.get("user_tags", []):
            counts[str(tag)] = counts.get(str(tag), 0) + 1
        for reason in item.get("quality_loss_reasons", [])[:3]:
            counts[str(reason)] = counts.get(str(reason), 0) + 1
    return [key for key, _count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:8]]


def _preference_summary(examples: list[dict[str, Any]]) -> str:
    good_count = len([item for item in examples if int(item.get("user_rating", 0) or 0) >= 4])
    bad_count = len([item for item in examples if int(item.get("user_rating", 0) or 0) in {1, 2}])
    return f"{good_count} high-rated examples, {bad_count} low-rated examples available."
