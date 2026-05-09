# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from pathlib import Path

from gmdgen.validation.code_validation import (
    CodeValidationResult,
    detect_optional_tool,
    run_code_validation_suite,
    run_command_safely,
)
from gmdgen.validation.runtime_audit import run_static_exception_audit


def test_code_validation_report_structure(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "guide.md").write_text("trigger schema", encoding="utf-8")

    monkeypatch.setattr(
        "gmdgen.validation.code_validation.run_command_safely",
        lambda command, **_kwargs: CodeValidationResult(command=list(command), passed=True),
    )
    monkeypatch.setattr("gmdgen.validation.code_validation.detect_optional_tool", lambda _name: False)
    monkeypatch.setattr("gmdgen.validation.code_validation.run_full_runtime_audit", lambda _root: {})

    report = run_code_validation_suite(tmp_path, dataset_dir=dataset, include_pytest=False)

    assert report.overall_passed is True
    assert report.results
    assert report.finished_at


def test_code_validation_skips_missing_optional_tools(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    monkeypatch.setattr(
        "gmdgen.validation.code_validation.run_command_safely",
        lambda command, **_kwargs: CodeValidationResult(command=list(command), passed=True),
    )
    monkeypatch.setattr("gmdgen.validation.code_validation.detect_optional_tool", lambda _name: False)
    monkeypatch.setattr("gmdgen.validation.code_validation.run_full_runtime_audit", lambda _root: {})

    report = run_code_validation_suite(tmp_path, dataset_dir=dataset, include_pytest=False)

    assert any(result.get("skipped") for result in report.results)
    assert any("not installed" in warning for warning in report.warnings)


def test_code_validation_redacts_secrets(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    def fake_run(command, **_kwargs):
        return CodeValidationResult(command=list(command), passed=True, stdout_tail="ok sk-secretsecret")

    monkeypatch.setattr("gmdgen.validation.code_validation.run_command_safely", fake_run)
    monkeypatch.setattr("gmdgen.validation.code_validation.detect_optional_tool", lambda _name: False)
    monkeypatch.setattr("gmdgen.validation.code_validation.run_full_runtime_audit", lambda _root: {})

    report = run_code_validation_suite(tmp_path, dataset_dir=dataset, include_pytest=False)

    assert "sk-secretsecret" not in str(report.to_dict())


def test_code_validation_can_run_compileall_command_mocked(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    seen: list[list[str]] = []

    def fake_run(command, **_kwargs):
        seen.append(list(command))
        return CodeValidationResult(command=list(command), passed=True)

    monkeypatch.setattr("gmdgen.validation.code_validation.run_command_safely", fake_run)
    monkeypatch.setattr("gmdgen.validation.code_validation.detect_optional_tool", lambda _name: False)
    monkeypatch.setattr("gmdgen.validation.code_validation.run_full_runtime_audit", lambda _root: {})

    run_code_validation_suite(tmp_path, dataset_dir=dataset, include_pytest=False)

    assert any("-m" in command and "compileall" in command for command in seen)


import sys

def test_run_command_safely_success(tmp_path: Path) -> None:
    result = run_command_safely([sys.executable, "-c", "print('ok')"], cwd=tmp_path, timeout=30)

    assert result.passed is True
    assert "ok" in result.stdout_tail


def test_detect_optional_tool_returns_bool() -> None:
    assert isinstance(detect_optional_tool("definitely-not-a-real-tool-name"), bool)


def test_static_exception_audit_skips_when_ruff_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("gmdgen.validation.runtime_audit.shutil.which", lambda _name: None)

    result = run_static_exception_audit(tmp_path)

    assert result["passed"] is True
    assert "skipped" in result["output"]
