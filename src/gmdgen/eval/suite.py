import json
from pathlib import Path
from typing import Any

from gmdgen.eval.cases import EvalCase
from gmdgen.eval.report import EvalResult
from gmdgen.generate.generator import generate_from_config

class QualityEvalSuite:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_case(self, case: EvalCase, is_live_ollama: bool = False, config_overrides: dict[str, Any] | None = None) -> EvalResult:
        config = {
            "audio_file": case.audio_fixture,
            "style_reference_level": case.reference_level_fixture or None,
            "difficulty": case.difficulty,
            "quality_mode": case.quality_mode,
            "output_dir": str(self.output_dir),
            "output_name": f"eval_{case.case_id}",
            "ai_provider": "ollama" if is_live_ollama else "local_test_only",
            "allow_local_test_provider": not is_live_ollama,
            "real_generation_requires_ollama": is_live_ollama,
            "save_validation_report": True,
            "save_learning_data": True,
            "learning_store_dir": str(self.output_dir / "learning"),
            "enforce_quality_gate": False,
        }
        if config_overrides:
            config.update(config_overrides)

        report_path = self.output_dir / f"eval_{case.case_id}_report.json"
        try:
            result = generate_from_config(config)
        except Exception as exc:
            eval_res = EvalResult(
                case_id=case.case_id,
                passed=False,
                failed_thresholds=[f"Exception during generation: {exc}"],
                report_path=str(report_path)
            )
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(eval_res.to_dict(), f, indent=2)
            return eval_res

        score_breakdown = result.get("score", {})
        quality_loss_reasons = result.get("quality_loss_reasons", [])
        snapshots = result.get("plan_snapshots", [])
        snapshot_summary = snapshots[-1] if snapshots else {}
        selected_candidate = result.get("selected_candidate_summary", {})

        failed = []
        if result.get("num_objects", 0) < case.expected_min_object_count:
            failed.append(f"num_objects {result.get('num_objects', 0)} < {case.expected_min_object_count}")
        
        # In a real impl we'd extract trigger_count, repair_loss_ratio, drop_impact etc.
        # This mocks the check for now but uses the dicts.
        drop_impact = score_breakdown.get("drop_impact", 0.0)
        if drop_impact < case.expected_min_drop_impact:
            failed.append(f"drop_impact {drop_impact} < {case.expected_min_drop_impact}")

        repair_loss = score_breakdown.get("repair_loss_ratio", 0.0)
        if repair_loss > case.expected_max_repair_loss:
            failed.append(f"repair_loss {repair_loss} > {case.expected_max_repair_loss}")


        eval_result = EvalResult(
            case_id=case.case_id,
            passed=len(failed) == 0,
            score_breakdown=score_breakdown,
            failed_thresholds=failed,
            quality_loss_reasons=quality_loss_reasons,
            selected_candidate_summary=selected_candidate,
            plan_snapshot_summary=snapshot_summary,
            learning_example_id=result.get("learning_example_id", ""),
            report_path=str(report_path)
        )

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(eval_result.to_dict(), f, indent=2)

        return eval_result
