from pathlib import Path
from gmdgen.dataset.bootstrap import initialize_dataset_structure
from gmdgen.dataset.reference_quality import evaluate_reference_levels

def test_dataset_bootstrap_creates_expected_dirs(tmp_path: Path):
    res = initialize_dataset_structure(tmp_path)
    assert len(res["created"]) > 0
    assert (tmp_path / "reference_levels" / "modern").exists()
    assert (tmp_path / "eval" / "reports").exists()

def test_reference_level_quality_report(tmp_path: Path):
    res = initialize_dataset_structure(tmp_path)
    # create a dummy valid gmd
    gmd_path = tmp_path / "reference_levels" / "modern" / "test.gmd"
    gmd_path.write_text("H4sIAAAAAAAACw... dummy base64 gz ...", encoding="utf-8")
    
    report = evaluate_reference_levels(tmp_path)
    # The dummy will fail to parse since it's not valid, which is fine, we just check it doesn't crash
    assert "invalid_count" in report
    assert report["invalid_count"] == 1

def test_invalid_reference_level_reported_not_crash(tmp_path: Path):
    initialize_dataset_structure(tmp_path)
    gmd_path = tmp_path / "reference_levels" / "modern" / "bad.gmd"
    gmd_path.write_text("not a gmd", encoding="utf-8")
    report = evaluate_reference_levels(tmp_path)
    assert report["invalid_count"] == 1
    assert "bad.gmd" in report["invalid_files"][0]["file"]
