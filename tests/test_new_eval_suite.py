import pytest
from pathlib import Path
from gmdgen.eval.suite import QualityEvalSuite
from gmdgen.eval.cases import EvalCase

def test_new_eval_suite_rejects_too_sparse(tmp_path: Path):
    suite = QualityEvalSuite(tmp_path)
    case = EvalCase(
        case_id="too_sparse_negative",
        audio_fixture="tests/fixtures/audio/clicks.wav",
        expected_min_object_count=9999,
    )
    # the local test provider should give 0 objects if no mock is set, or crash
    # in any case it should fail
    res = suite.run_case(case, is_live_ollama=False)
    assert res.passed is False

def test_new_eval_report_serializes(tmp_path: Path):
    suite = QualityEvalSuite(tmp_path)
    case = EvalCase(
        case_id="serializes",
        audio_fixture="tests/fixtures/audio/clicks.wav",
        expected_min_object_count=0, # impossible to fail
    )
    res = suite.run_case(case, is_live_ollama=False)
    assert Path(res.report_path).exists()
    
    import json
    data = json.loads(Path(res.report_path).read_text())
    assert data["case_id"] == "serializes"
