from pathlib import Path
from gmdgen.output.bundle import package_output_bundle

def test_output_bundle_contains_expected_files(tmp_path: Path):
    res = {
        "output_path": "non_existent.gmd",
        "validation_report": {"score": 1.0, "api_key": "sk-123"},
        "score": {"total": 1.0},
        "plan_snapshots": [{"stage": "raw"}],
        "debug_artifacts": {"ollama": {"key": "sk-456"}},
        "valid": True
    }
    
    bundle_dir = package_output_bundle(res, tmp_path, "my_bundle", include_debug=True)
    assert bundle_dir.exists()
    assert (bundle_dir / "validation_report.json").exists()
    assert (bundle_dir / "quality_report.json").exists()
    assert (bundle_dir / "plan_snapshots" / "snapshot_0.json").exists()
    assert (bundle_dir / "debug_artifacts" / "ollama.json").exists()
    assert (bundle_dir / "README_generated.txt").exists()

def test_output_bundle_excludes_api_key(tmp_path: Path):
    res = {
        "validation_report": {"api_key": "sk-123", "nested": {"api_key": "sk-456", "val": "sk-789"}},
        "debug_artifacts": {"ai": {"text": "Bearer sk-abc"}}
    }
    bundle_dir = package_output_bundle(res, tmp_path, "secure_bundle", include_debug=True)
    
    val_text = (bundle_dir / "validation_report.json").read_text()
    assert "sk-123" not in val_text
    assert "sk-456" not in val_text
    assert "sk-[REDACTED]" in val_text
    
    dbg_text = (bundle_dir / "debug_artifacts" / "ai.json").read_text()
    assert "sk-abc" not in dbg_text
    assert "sk-[REDACTED]" in dbg_text
