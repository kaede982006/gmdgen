# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def export_fine_tuning_examples(examples: Iterable[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            _validate_example(example)
            sanitized = _strip_sensitive(example)
            handle.write(json.dumps(sanitized, ensure_ascii=False, sort_keys=True) + "\n")


def export_high_quality_fine_tuning_examples(
    examples: Iterable[dict[str, Any]],
    output_path: str | Path,
    *,
    min_rating: int = 4,
    min_score: float = 0.65,
    max_repair_loss: float = 0.35,
) -> int:
    selected = []
    for example in examples:
        if _quality_eligible(example, min_rating=min_rating, min_score=min_score, max_repair_loss=max_repair_loss):
            selected.append(example)
    export_fine_tuning_examples(selected, output_path)
    return len(selected)


def build_example_from_generation_run(result: dict[str, Any], *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or {}
    validation = result.get("validation_report", {}) if isinstance(result.get("validation_report", {}), dict) else {}
    return {
        "input": {
            "audio_summary": {
                "audio_file_name": result.get("audio_file_name", ""),
                "audio_backend": result.get("audio_backend", ""),
                "detected_bpm": result.get("detected_bpm", result.get("bpm", 0.0)),
                "beat_count": result.get("beat_count", result.get("num_beats", 0)),
                "onset_count": result.get("onset_count", result.get("num_onsets", 0)),
                "section_count": result.get("section_count", result.get("num_sections", 0)),
            },
            "section_plans": result.get("section_plan", []),
            "time_x_summary": result.get("time_x_report", {}),
            "style_reference_summary": result.get("style_reference_summary", {}),
            "difficulty": config.get("difficulty", "normal"),
            "object_budget": config.get("object_budget", config.get("num_objects", 0)),
            "safe_mode": config.get("safe_mode", True),
        },
        "output": {
            "sections": result.get("section_plan", []),
            "object_plans": result.get("ai_output_preview", {}).get("object_plans", []),
            "trigger_plans": result.get("ai_output_preview", {}).get("trigger_plans", []),
            "validation_score": validation.get("score", result.get("final_score", 0.0)),
            "safety_notes": validation.get("warnings", []),
        },
    }


def validate_fine_tuning_jsonl(path: str | Path) -> list[str]:
    issues: list[str] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            _validate_example(payload)
        except Exception as exc:  # noqa: BLE001
            issues.append(f"line {line_no}: {exc}")
    return issues


def _validate_example(example: dict[str, Any]) -> None:
    if not isinstance(example, dict):
        raise TypeError("fine-tuning example must be a mapping")
    if "input" not in example or "output" not in example:
        raise ValueError("fine-tuning example requires input and output")
    output = example["output"]
    if not isinstance(output, dict):
        raise ValueError("fine-tuning output must be a mapping")
    if "raw_save_string" in output or "level_string" in output:
        raise ValueError("fine-tuning export must use structured plans, not raw save strings")


def _strip_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _strip_sensitive(item)
            for key, item in value.items()
            if "api_key" not in str(key).lower()
            and "base_url" not in str(key).lower()
            and "host" not in str(key).lower()
        }
    if isinstance(value, list):
        return [_strip_sensitive(item) for item in value]
    if isinstance(value, str) and value.startswith("sk-"):
        return "[REDACTED]"
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        if ":" in normalized[:4] and "/" in normalized:
            return Path(normalized).name
    return value


def _quality_eligible(example: dict[str, Any], *, min_rating: int, min_score: float, max_repair_loss: float) -> bool:
    output = example.get("output", {}) if isinstance(example, dict) else {}
    validation_score = float(output.get("validation_score", output.get("score", 0.0)) or 0.0)
    repair_loss = float(output.get("repair_loss_ratio", 0.0) or 0.0)
    rating = int(output.get("user_rating", example.get("user_rating", 0)) or 0)
    editor_safety = float(output.get("editor_safety_score", 1.0) or 1.0)
    return validation_score >= min_score and repair_loss <= max_repair_loss and editor_safety >= 0.75 and rating >= min_rating
