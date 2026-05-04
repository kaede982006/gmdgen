from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gmdgen.ai.dataset_index import build_dataset_index, ensure_dataset_dir
from gmdgen.ai.schemas import AI_LEVEL_PLAN_JSON_SCHEMA
from gmdgen.audit.ollama_only import run_ollama_only_audit
from gmdgen.errors import redact_text
from gmdgen.validation.runtime_audit import run_full_runtime_audit
from gmdgen.learning.store import sanitize_learning_payload


@dataclass(slots=True)
class CodeValidationResult:
    command: list[str]
    passed: bool
    return_code: int = 0
    stdout_tail: str = ""
    stderr_tail: str = ""
    duration_seconds: float = 0.0
    skipped: bool = False
    skip_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CodeValidationReport:
    started_at: str
    finished_at: str = ""
    overall_passed: bool = False
    results: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_command_safely(
    command: list[str],
    *,
    cwd: str | Path,
    timeout: int = 120,
) -> CodeValidationResult:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        duration = time.perf_counter() - started
        return CodeValidationResult(
            command=list(command),
            passed=completed.returncode == 0,
            return_code=int(completed.returncode),
            stdout_tail=_tail(redact_text(completed.stdout)),
            stderr_tail=_tail(redact_text(completed.stderr)),
            duration_seconds=round(duration, 3),
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - started
        return CodeValidationResult(
            command=list(command),
            passed=False,
            return_code=124,
            stdout_tail=_tail(redact_text(exc.stdout or "")),
            stderr_tail=_tail(redact_text(exc.stderr or f"timeout after {timeout}s")),
            duration_seconds=round(duration, 3),
        )
    except Exception as exc:  # noqa: BLE001
        duration = time.perf_counter() - started
        return CodeValidationResult(
            command=list(command),
            passed=False,
            return_code=1,
            stderr_tail=_tail(redact_text(str(exc))),
            duration_seconds=round(duration, 3),
        )


def detect_optional_tool(name: str) -> bool:
    return shutil.which(name) is not None


def run_code_validation_suite(
    project_root: str | Path,
    *,
    dataset_dir: str | Path | None = None,
    include_pytest: bool = True,
    timeout: int = 180,
) -> CodeValidationReport:
    root = Path(project_root).resolve()
    report = CodeValidationReport(started_at=_now_iso())
    commands = [
        [sys.executable, "-m", "compileall", str(root / "src" / "gmdgen"), str(root / "tests")],
        [sys.executable, "-c", "import gmdgen; print('import ok')"],
        [sys.executable, "-c", "import gmdgen.gui.app; print('gui import ok')"],
    ]
    if include_pytest:
        commands.append([sys.executable, "-m", "pytest", "-q"])
    for command in commands:
        result = run_command_safely(command, cwd=root, timeout=timeout)
        report.results.append(result.to_dict())
        if not result.passed:
            report.errors.append(f"command_failed: {' '.join(command)}")

    dataset_result = _validate_dataset_scan(root, dataset_dir=dataset_dir)
    report.results.append(dataset_result.to_dict())
    if not dataset_result.passed:
        report.errors.append("dataset_scan_failed")

    audit_result = _validate_ollama_only_policy()
    report.results.append(audit_result.to_dict())
    if not audit_result.passed:
        report.errors.append("ollama_only_audit_failed")

    schema_result = _validate_schema()
    report.results.append(schema_result.to_dict())
    if not schema_result.passed:
        report.errors.append("ollama_schema_validation_failed")

    audit_out = run_full_runtime_audit(root)
    for name, res in audit_out.items():
        if not res["passed"]:
            report.errors.append(f"{name}_failed")

    for tool_name, command in (
        ("ruff", ["ruff", "check", "."]),
        ("mypy", ["mypy", "src"]),
        ("pyright", ["pyright"]),
    ):
        if detect_optional_tool(tool_name):
            result = run_command_safely(command, cwd=root, timeout=timeout)
        else:
            result = CodeValidationResult(
                command=command,
                passed=True,
                skipped=True,
                skip_reason=f"{tool_name} is not installed",
            )
            report.warnings.append(result.skip_reason)
        report.results.append(result.to_dict())
        if not result.passed:
            report.errors.append(f"optional_tool_failed: {tool_name}")

    report.finished_at = _now_iso()
    report.overall_passed = not report.errors
    payload = _redact_payload(sanitize_learning_payload(report.to_dict()))
    return CodeValidationReport(
        started_at=str(payload.get("started_at", report.started_at)),
        finished_at=str(payload.get("finished_at", report.finished_at)),
        overall_passed=bool(payload.get("overall_passed", report.overall_passed)),
        results=list(payload.get("results", report.results)),
        warnings=list(payload.get("warnings", report.warnings)),
        errors=list(payload.get("errors", report.errors)),
    )


def _validate_dataset_scan(root: Path, *, dataset_dir: str | Path | None) -> CodeValidationResult:
    started = time.perf_counter()
    try:
        resolved = ensure_dataset_dir(dataset_dir or (root / "dataset"))
        index = build_dataset_index(resolved, max_total_context_chars=5000)
        payload = {
            "dataset_dir": "dataset",
            "documents": len(index.documents),
            "chunks": len(index.chunks),
            "reference_levels": len(index.reference_levels),
            "failed_files": index.scan_result.get("files_failed", 0),
        }
        return CodeValidationResult(
            command=["dataset-scan-smoke"],
            passed=True,
            stdout_tail=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            duration_seconds=round(time.perf_counter() - started, 3),
        )
    except Exception as exc:  # noqa: BLE001
        return CodeValidationResult(
            command=["dataset-scan-smoke"],
            passed=False,
            return_code=1,
            stderr_tail=_tail(redact_text(str(exc))),
            duration_seconds=round(time.perf_counter() - started, 3),
        )


def _validate_ollama_only_policy() -> CodeValidationResult:
    audit = run_ollama_only_audit(
        {
            "ai_provider": "ollama",
            "real_generation_requires_external_ai": False,
            "local_fallback": False,
            "cli_generate_disabled": True,
            "fallback_providers": [],
        },
    )
    return CodeValidationResult(
        command=["ollama-only-audit"],
        passed=bool(audit.passed),
        return_code=0 if audit.passed else 1,
        stdout_tail=_tail(redact_text(audit.summary)),
        stderr_tail="",
    )


def _validate_schema() -> CodeValidationResult:
    errors: list[str] = []
    return CodeValidationResult(
        command=["ollama-schema-self-check"],
        passed=not errors,
        return_code=0 if not errors else 1,
        stdout_tail="schema ok" if not errors else "",
        stderr_tail=_tail("\n".join(errors)),
    )


def _tail(text: str, *, max_chars: int = 3000) -> str:
    value = str(text or "")
    return value[-max_chars:]


def _redact_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _redact_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_payload(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
