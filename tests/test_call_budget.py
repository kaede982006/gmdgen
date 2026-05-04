# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import pytest

from gmdgen.ai.cache import AIRequestBudget


def test_call_budget_counts_provider_calls() -> None:
    budget = AIRequestBudget(max_calls=2)

    budget.record_call("ollama")
    budget.record_call("ollama")

    assert budget.calls_used == 2
    assert budget.provider_calls["ollama"] == 2


def test_call_budget_blocks_excess_calls() -> None:
    budget = AIRequestBudget(max_calls=1)
    budget.record_call("ollama")

    with pytest.raises(RuntimeError, match="AI call budget exhausted"):
        budget.record_call("ollama")
    assert budget.stopped_by_budget is True
