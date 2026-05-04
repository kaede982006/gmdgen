# SPDX-License-Identifier: GPL-3.0-or-later
import os
from pathlib import Path

from gmdgen.eval.cases import EvalCase
from gmdgen.eval.suite import QualityEvalSuite

def run_live_eval(output_dir: Path) -> dict[str, bool]:
    if os.environ.get("RUN_OLLAMA_LIVE_TESTS") != "1":
        return {"skipped": True, "reason": "RUN_OLLAMA_LIVE_TESTS != 1"}
    if not os.environ.get("OLLAMA_HOST"):
        return {"skipped": True, "reason": "OLLAMA_HOST is not set"}

    suite = QualityEvalSuite(output_dir)
    # Using a generated audio file from our tests or a dummy path
    case = EvalCase(
        case_id="live_synthetic_120bpm",
        audio_fixture="tests/fixtures/audio/clicks.wav",
        expected_min_object_count=5,
        expected_min_drop_impact=0.0
    )

    result = suite.run_case(case, is_live_ollama=True)
    return {"skipped": False, "passed": result.passed, "report_path": result.report_path}
