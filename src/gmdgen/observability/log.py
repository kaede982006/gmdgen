# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 kaede982006
"""Structured JSONL + human-friendly logger.

Single source of truth for runtime telemetry. Every line carries a
canonical schema (ts, run_id, phase, step, event, payload, level,
elapsed_ms, mem_mb) so logs are queryable post-hoc.

Usage:
    log = get_logger()
    log.event(phase="plan", step="build", event="phase_start")

    @logged(phase="plan", step="build")
    def build_level_plan(...):
        ...
"""
from __future__ import annotations

import json
import os
import secrets
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator


class LogLevel(IntEnum):
    QUIET = 0
    NORMAL = 1
    VERBOSE = 2
    TRACE = 3


# Global state — kept tiny and explicit.
_LEVEL: LogLevel = LogLevel.NORMAL
_LOGGER: "StructuredLogger | None" = None


def set_log_level(level: int | LogLevel) -> None:
    global _LEVEL
    _LEVEL = LogLevel(int(level))


def _resolve_level() -> LogLevel:
    raw = os.environ.get("GMDGEN_LOG_LEVEL")
    if raw is None:
        return _LEVEL
    try:
        return LogLevel(int(raw))
    except (ValueError, KeyError):
        return _LEVEL


def _default_log_dir() -> Path:
    base = os.environ.get("GMDGEN_CACHE_DIR")
    if base:
        return Path(base) / "logs"
    return Path.home() / ".cache" / "gmdgen" / "logs"


def _fallback_log_dirs() -> list[Path]:
    return [
        Path.cwd() / ".gmdgen" / "logs",
        Path(tempfile.gettempdir()) / "gmdgen" / "logs",
    ]


def _mem_mb() -> float:
    """Return current process memory in MB (best-effort, cross-platform)."""
    try:
        import resource  # POSIX only
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is KB on Linux, bytes on macOS
        if sys.platform == "darwin":
            return rusage.ru_maxrss / (1024 * 1024)
        return rusage.ru_maxrss / 1024
    except ImportError:
        try:
            import psutil  # type: ignore
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0


@dataclass
class StructuredLogger:
    """Writes JSONL to a file and human-friendly lines to stderr."""

    run_id: str
    log_path: Path
    level: LogLevel = LogLevel.NORMAL
    started_at: float = field(default_factory=time.time)
    _file: Any = None

    def __post_init__(self) -> None:
        candidates = [self.log_path, *[path / self.log_path.name for path in _fallback_log_dirs()]]
        last_exc: Exception | None = None
        for candidate in candidates:
            try:
                candidate.parent.mkdir(parents=True, exist_ok=True)
                self._file = candidate.open("a", encoding="utf-8")
                self.log_path = candidate
                return
            except OSError as exc:
                last_exc = exc
                continue
        # Telemetry must never break generation. Fall back to a sink.
        self._file = open(os.devnull, "a", encoding="utf-8")
        if last_exc is not None and int(self.level) >= int(LogLevel.NORMAL):
            print(f"[observability] log_open_fallback error={last_exc}", file=sys.stderr)

    def event(
        self,
        *,
        phase: str,
        step: str,
        event: str,
        payload: dict[str, Any] | None = None,
        level: LogLevel | int = LogLevel.NORMAL,
    ) -> None:
        """Emit one structured event."""
        if int(level) > int(self.level):
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "phase": phase,
            "step": step,
            "event": event,
            "payload": payload or {},
            "level": int(level),
            "elapsed_ms": int((time.time() - self.started_at) * 1000),
            "mem_mb": round(_mem_mb(), 1),
        }
        try:
            self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._file.flush()
        except Exception:
            # Telemetry must never break generation.
            pass
        # Human-friendly tail to stderr at NORMAL+
        if int(self.level) >= int(LogLevel.NORMAL):
            extras = ""
            if payload:
                extras = " " + " ".join(f"{k}={v}" for k, v in payload.items())
            print(f"[{phase}] {step:<10} {event}{extras}", file=sys.stderr)

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass


def get_logger() -> StructuredLogger:
    global _LOGGER
    if _LOGGER is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_id = f"{ts}-{secrets.token_hex(4)}"
        log_dir = _default_log_dir()
        _LOGGER = StructuredLogger(
            run_id=run_id,
            log_path=log_dir / f"{run_id}.jsonl",
            level=_resolve_level(),
        )
    return _LOGGER


def logged(*, phase: str, step: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: emits phase_start and phase_end with elapsed_ms.

    Exceptions are recorded as event="phase_error" and re-raised.
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        actual_step = step or fn.__name__

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            log = get_logger()
            t0 = time.time()
            log.event(phase=phase, step=actual_step, event="phase_start")
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                log.event(
                    phase=phase, step=actual_step, event="phase_error",
                    payload={"error": type(exc).__name__, "message": str(exc)[:200]},
                )
                raise
            log.event(
                phase=phase, step=actual_step, event="phase_end",
                payload={"duration_ms": int((time.time() - t0) * 1000)},
            )
            return result
        return wrapper
    return decorator


@contextmanager
def session(run_id: str | None = None) -> Iterator[StructuredLogger]:
    """Context manager that creates and tears down a logger session."""
    global _LOGGER
    prev = _LOGGER
    if run_id is not None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        rid = f"{ts}-{run_id}"
        _LOGGER = StructuredLogger(
            run_id=rid,
            log_path=_default_log_dir() / f"{rid}.jsonl",
            level=_resolve_level(),
        )
    log = get_logger()
    try:
        yield log
    finally:
        log.close()
        _LOGGER = prev
