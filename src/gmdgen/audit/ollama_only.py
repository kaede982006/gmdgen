# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

@dataclass
class OllamaOnlyAuditCheck:
    """Audit check result for Ollama-only policy."""
    name: str
    passed: bool
    importance: str = "critical"
    message: str = ""
    recommendation: str = ""

@dataclass
class OllamaOnlyAuditResult:
    """Audit result for Ollama-only policy."""
    passed: bool
    summary: str
    checks: list[OllamaOnlyAuditCheck] = field(default_factory=list)
    ollama_provider_available: bool = True
    default_provider: str = "ollama"

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "summary": self.summary,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "importance": c.importance,
                    "message": c.message,
                    "recommendation": c.recommendation,
                }
                for c in self.checks
            ],
            "ollama_provider_available": self.ollama_provider_available,
            "default_provider": self.default_provider,
        }

def run_ollama_only_audit(config: dict[str, Any], app_state: dict[str, Any] | None = None) -> OllamaOnlyAuditResult:
    checks: list[OllamaOnlyAuditCheck] = []
    checks.extend(audit_generation_config(config))
    passed = all(c.passed for c in checks if c.importance == "critical")
    summary = "Ollama-only generation policy passed" if passed else "Ollama-only generation policy failed"
    return OllamaOnlyAuditResult(
        passed=passed,
        summary=summary,
        checks=checks,
        ollama_provider_available=True,
        default_provider="ollama",
    )

def audit_generation_config(config: dict[str, Any]) -> list[OllamaOnlyAuditCheck]:
    provider = str(config.get("ai_provider", "ollama")).strip().lower()
    return [
        OllamaOnlyAuditCheck(
            name="ai_provider_is_ollama",
            passed=provider == "ollama",
            importance="critical",
            message=f"ai_provider must be 'ollama' (got '{provider}')",
            recommendation="Set ai_provider='ollama'.",
        )
    ]
