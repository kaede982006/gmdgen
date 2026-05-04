from __future__ import annotations

from pathlib import Path

from gmdgen.eval.learning_memory_health import analyze_learning_memory_health


def test_learning_memory_health_report(tmp_path: Path) -> None:
    report = analyze_learning_memory_health(tmp_path)
    assert report.total_examples == 0
    assert isinstance(report.to_dict(), dict)
