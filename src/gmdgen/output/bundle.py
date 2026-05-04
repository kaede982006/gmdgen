import json
import shutil
from pathlib import Path
from typing import Any

from gmdgen.gui.app import sanitize_report

def package_output_bundle(
    result: dict[str, Any],
    output_dir: str | Path,
    bundle_name: str,
    include_debug: bool = False
) -> Path:
    base = Path(output_dir) / bundle_name
    base.mkdir(parents=True, exist_ok=True)
    
    # 1. Level file
    if "output_path" in result and Path(result["output_path"]).exists():
        ext = Path(result["output_path"]).suffix
        shutil.copy2(result["output_path"], base / f"level{ext}")
        
    # 2. Validation report
    if "validation_report" in result:
        report_path = base / "validation_report.json"
        report_path.write_text(json.dumps(sanitize_report(result["validation_report"]), indent=2), encoding="utf-8")
        
    # 3. Quality report
    if "score" in result:
        q_report = {
            "score_breakdown": result.get("score", {}),
            "quality_loss_reasons": result.get("quality_loss_reasons", []),
            "geode_checked": result.get("geode_checked", False)
        }
        (base / "quality_report.json").write_text(json.dumps(q_report, indent=2), encoding="utf-8")
        
    # 4. Plan snapshots
    if "plan_snapshots" in result:
        snaps_dir = base / "plan_snapshots"
        snaps_dir.mkdir(exist_ok=True)
        for i, snap in enumerate(result["plan_snapshots"]):
            (snaps_dir / f"snapshot_{i}.json").write_text(json.dumps(snap, indent=2), encoding="utf-8")

    # 5. Debug artifacts
    if include_debug and "debug_artifacts" in result:
        dbg_dir = base / "debug_artifacts"
        dbg_dir.mkdir(exist_ok=True)
        for k, v in result["debug_artifacts"].items():
            (dbg_dir / f"{k}.json").write_text(json.dumps(sanitize_report(v), indent=2), encoding="utf-8")

    # 6. README
    readme = [
        f"Generated Bundle: {bundle_name}",
        f"Generated At: {result.get('generated_at', 'unknown')}",
        f"Model: {result.get('ai_model', 'unknown')}",
        f"Quality Gate Passed: {result.get('valid', False)}",
        f"Final Score: {result.get('score', {}).get('total', 0.0)}",
    ]
    (base / "README_generated.txt").write_text("\n".join(readme), encoding="utf-8")
    
    return base
