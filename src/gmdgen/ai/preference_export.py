# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from gmdgen.feedback.store import sanitize_feedback_payload


def export_preference_pairs(pairs: Iterable[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            sanitized = sanitize_feedback_payload(pair)
            _validate_pair(sanitized)
            handle.write(json.dumps(sanitized, ensure_ascii=False, sort_keys=True) + "\n")


def build_preference_pair(
    *,
    input_summary: dict[str, Any],
    chosen_candidate: dict[str, Any],
    rejected_candidate: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "input": dict(input_summary),
        "chosen": dict(chosen_candidate),
        "rejected": dict(rejected_candidate),
        "reason": str(reason)[:500],
    }


def validate_preference_jsonl(path: str | Path) -> list[str]:
    issues = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            _validate_pair(json.loads(line))
        except Exception as exc:  # noqa: BLE001
            issues.append(f"line {line_no}: {exc}")
    return issues


def _validate_pair(pair: dict[str, Any]) -> None:
    if "chosen" not in pair or "rejected" not in pair or "input" not in pair:
        raise ValueError("preference pair requires input, chosen, and rejected")
    lowered = json.dumps(pair).lower()
    if "api_key" in lowered or "base_url" in lowered or "host" in lowered:
        raise ValueError("preference pair must not include sensitive fields (api_key, base_url, host)")
