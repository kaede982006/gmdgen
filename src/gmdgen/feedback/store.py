# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class FeedbackRecord:
    input_summary: dict[str, Any] = field(default_factory=dict)
    audio_summary: dict[str, Any] = field(default_factory=dict)
    reference_summary: dict[str, Any] = field(default_factory=dict)
    prompt_summary: str = ""
    selected_candidate_plan: dict[str, Any] = field(default_factory=dict)
    final_plan: dict[str, Any] = field(default_factory=dict)
    score_breakdown: dict[str, Any] = field(default_factory=dict)
    user_rating: int = 0
    user_tags: list[str] = field(default_factory=list)
    user_notes: str = ""
    debug_artifact_links: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_feedback_record(record: FeedbackRecord | dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = sanitize_feedback_payload(record.to_dict() if isinstance(record, FeedbackRecord) else dict(record))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def load_feedback_records(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    records = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def export_feedback_eval_dataset(feedback_path: str | Path, output_path: str | Path, *, min_rating: int = 4) -> None:
    records = [
        record
        for record in load_feedback_records(feedback_path)
        if int(record.get("user_rating", 0) or 0) >= min_rating
    ]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "input": {
                "audio_summary": record.get("audio_summary", {}),
                "reference_summary": record.get("reference_summary", {}),
                "prompt_summary": record.get("prompt_summary", ""),
            },
            "output": {
                "final_plan": record.get("final_plan", {}),
                "score_breakdown": record.get("score_breakdown", {}),
                "user_rating": record.get("user_rating", 0),
                "user_tags": record.get("user_tags", []),
            },
        }
        for record in records
    ]
    path.write_text(json.dumps(sanitize_feedback_payload(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def low_rated_feedback_context(feedback_path: str | Path, *, max_items: int = 5) -> list[str]:
    records = load_feedback_records(feedback_path)
    result = []
    for record in records:
        if int(record.get("user_rating", 0) or 0) <= 2:
            tags = ", ".join(record.get("user_tags", [])[:6])
            note = str(record.get("user_notes", ""))[:180]
            result.append(f"Previous low-rated output tags=[{tags}] note={note}")
    return result[-max_items:]


def sanitize_feedback_payload(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            if "api_key" in lowered or "ollama_key" in lowered or "base_url" in lowered or "host" in lowered:
                continue
            sanitized[key_text] = sanitize_feedback_payload(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_feedback_payload(item) for item in value]
    if isinstance(value, str):
        if value.startswith("sk-"):
            return "sk-[REDACTED]"
        text = value.replace("\\", "/")
        if ":" in text[:4] and "/" in text:
            return Path(text).name
        return value
    return value
