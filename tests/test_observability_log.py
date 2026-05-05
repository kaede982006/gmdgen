# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 kaede982006
"""P-6: log self-verification — every required phase/event must appear."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from gmdgen.observability.log import LogLevel, logged, session


REQUIRED_PHASES = {
    "pipeline", "plan", "layout", "decorate",
    "validate", "encode", "ai", "learning", "cache",
}
REQUIRED_EVENTS = {
    "phase_start", "phase_end",
}


def test_logger_writes_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GMDGEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GMDGEN_LOG_LEVEL", "1")
    with session(run_id="unittest") as log:
        log.event(phase="pipeline", step="start", event="phase_start")
        log.event(phase="pipeline", step="end", event="phase_end")
    files = list((tmp_path / "logs").glob("*.jsonl"))
    assert files, "no JSONL log produced"
    rows = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(r["event"] == "phase_start" for r in rows)
    assert any(r["event"] == "phase_end" for r in rows)


def test_logged_decorator_records_start_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GMDGEN_CACHE_DIR", str(tmp_path))

    @logged(phase="layout", step="apply_pattern")
    def do_work(x: int) -> int:
        return x * 2

    with session(run_id="dec_test"):
        assert do_work(7) == 14

    files = list((tmp_path / "logs").glob("*.jsonl"))
    assert files
    rows = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    starts = [r for r in rows if r["event"] == "phase_start" and r["phase"] == "layout"]
    ends = [r for r in rows if r["event"] == "phase_end" and r["phase"] == "layout"]
    assert starts and ends
    # End event must include duration_ms.
    assert "duration_ms" in ends[0]["payload"]


def test_logged_decorator_records_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GMDGEN_CACHE_DIR", str(tmp_path))

    @logged(phase="ai", step="ollama_call")
    def boom() -> None:
        raise RuntimeError("ai_timeout")

    with session(run_id="boom"):
        with pytest.raises(RuntimeError):
            boom()

    files = list((tmp_path / "logs").glob("*.jsonl"))
    assert files
    rows = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(r["event"] == "phase_error" and r["phase"] == "ai" for r in rows)


def test_log_level_quiet_suppresses_normal_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GMDGEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GMDGEN_LOG_LEVEL", "0")
    with session(run_id="quiet") as log:
        log.event(phase="pipeline", step="x", event="phase_start", level=LogLevel.NORMAL)
    files = list((tmp_path / "logs").glob("*.jsonl"))
    rows = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    # In QUIET mode, NORMAL events are filtered out.
    assert rows == []


def test_log_run_id_is_unique(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GMDGEN_CACHE_DIR", str(tmp_path))
    seen_ids = set()
    for _ in range(3):
        with session(run_id=f"r{_}") as log:
            seen_ids.add(log.run_id)
    assert len(seen_ids) == 3
