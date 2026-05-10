# SPDX-License-Identifier: GPL-3.0-or-later
import sys
import os
import argparse
import logging
import json
import uuid
import random
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

from gmdgen.ai.factory import create_ai_provider_from_config
from gmdgen.generate.generator import generate_from_config
from gmdgen.utils.seed import set_global_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="gmdgen: Geometry Dash AI Level Generator (Gemini-first CLI)",
        add_help=True
    )
    
    # Global provider options
    parser.add_argument("--provider", default="gemini", choices=["gemini"], help="AI provider (only gemini is supported)")
    parser.add_argument("--ai-provider", dest="provider", choices=["gemini"], help="(alias) AI provider")
    parser.add_argument("--model", help="Model name (e.g., gemini-2.5-flash)")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY", help="Environment variable for API key")
    
    # No fallback, no ollama, no qwen
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=2, help="Max retries")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--debug-provider", action="store_true", help="Debug provider calls")
    parser.add_argument("--allow-low-quality-draft", action="store_true", help="Allow saving low quality drafts")
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce console output (full logs saved to run.log)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    subparsers = parser.add_subparsers(dest="command", required=False)

    # Doctor
    doc_p = subparsers.add_parser("doctor", help="Check environment and configuration")
    doc_p.add_argument("--check-provider-live", action="store_true", help="Perform a live API smoke test")
    doc_p.add_argument("--provider", default="gemini", choices=["gemini"], help="AI provider")

    # Train
    train_p = subparsers.add_parser("train", help="Build dataset context")
    train_p.add_argument("--dataset", default="dataset", help="Path to dataset directory")
    train_p.add_argument("--force-rebuild", action="store_true", help="Force rebuild of dataset index")

    # Generate
    gen_p = subparsers.add_parser("generate", help="Generate a new level")
    gen_p.add_argument("--audio-file", "--audio", dest="audio_file", help="Path to audio file for conditioning")
    gen_p.add_argument("--config", dest="config", help="Path to generation config file (YAML)")
    gen_p.add_argument("--test-local-provider", dest="test_local_provider", action="store_true", help="Run generate with local test provider and print JSON to stdout (testing)")
    gen_p.add_argument("--output", "--output-dir", dest="output_dir", default="outputs", help="Output directory")
    gen_p.add_argument("--dry-run", action="store_true", help="Run generation without saving final files")
    
    # Exposing hidden but implemented features
    gen_p.add_argument("--object-count", type=int, help="Target number of objects (default: 1200)")
    gen_p.add_argument("--min-objects", type=int, help="Minimum number of objects (default: 100)")
    gen_p.add_argument("--max-objects", type=int, help="Maximum number of objects (default: 40000)")
    gen_p.add_argument("--duration", "--length", type=float, help="Target duration in seconds")
    gen_p.add_argument("--difficulty", type=float, help="Difficulty value [0.0 - 1.0] (default: 0.5)")
    gen_p.add_argument("--seed", type=int, help="Random seed for generation")
    gen_p.add_argument("--no-repair", action="store_false", dest="repair_level", help="Disable automatic structure repair")
    gen_p.set_defaults(repair_level=True)

    # Validate
    val_p = subparsers.add_parser("validate", help="Validate a level's structure and playability")
    val_p.add_argument("input", help="Input .gmd file or level string")
    val_p.add_argument("--output", help="Path to save validation report (JSON)")

    # Repair
    rep_p = subparsers.add_parser("repair", help="Repair common level structure issues")
    rep_p.add_argument("input", help="Input .gmd file")
    rep_p.add_argument("--output", help="Output path for repaired .gmd file")
    rep_p.add_argument("--report", help="Path to save repair report (JSON)")

    # Report
    rpt_p = subparsers.add_parser("report", help="Generate a human-readable quality report")
    rpt_p.add_argument("input", help="Input generation_report.json")
    rpt_p.add_argument("--output", help="Output path for human-readable report")

    return parser


class CLILogger:
    def __init__(self, run_id: str, output_dir: Path, is_quiet: bool = False, is_debug: bool = False):
        self.run_id = run_id
        self.output_dir = output_dir
        self.is_quiet = is_quiet
        self.is_debug = is_debug
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "run.log"
        self.events_file = self.output_dir / "events.jsonl"

    def log_event(self, category: str, stage: str, message: str, progress: int = 0, level: str = "INFO", **kwargs):
        if level == "DEBUG" and not self.is_debug:
            return
            
        ts = datetime.now().strftime("%H:%M:%S")
        rendered_line = f"[{ts}] [{category}] [{progress}%] {message}"
        
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "run_id": self.run_id,
            "command": kwargs.get("command", "unknown"),
            "category": category,
            "stage": stage,
            "level": level,
            "progress_percent": progress,
            "message": message,
            "rendered_line": rendered_line
        }
        
        with open(self.events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
            
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(rendered_line + "\n")
            
        # Console output: suppressed only if quite mode and it's not an ERROR.
        if not self.is_quiet or level == "ERROR":
            print(rendered_line)


def get_run_id_and_dir(output_base: str = "outputs"):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    uid = uuid.uuid4().hex[:6]
    run_id = f"{timestamp}_{uid}"
    output_dir = Path(output_base) / "runs" / run_id
    return run_id, output_dir

def cmd_doctor(args):
    print("=== gmdgen doctor ===")
    print(f"Python: {sys.version}")
    print(f"Root: {Path.cwd()}")
    
    api_key_env = args.api_key_env or "GEMINI_API_KEY"
    api_key = os.environ.get(api_key_env)
    
    if api_key:
        print(f"[OK] {api_key_env} found")
    else:
        print(f"[ERR] {api_key_env} missing. Set it with export {api_key_env}='your-key'")
    
    try:
        from google import genai
        print("[OK] google-genai installed")
    except ImportError:
        print("[ERR] google-genai missing. Run 'pip install google-genai'")

    if getattr(args, "check_provider_live", False):
        if not api_key:
            print("[ERR] Cannot perform live check: API key missing")
            sys.exit(1)
        print("[INFO] Performing live smoke test...")
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=args.model or "gemini-2.5-flash",
                contents="Say OK",
            )
            if response and response.text:
                print(f"[OK] Live check passed (model: {args.model or 'gemini-2.5-flash'})")
            else:
                print("[ERR] Live check returned empty response")
                sys.exit(1)
        except Exception as e:
            print(f"[ERR] Live check failed: {e}")
            sys.exit(1)

def cmd_train(args):
    run_id, output_dir = get_run_id_and_dir()
    cli_logger = CLILogger(run_id, output_dir, is_quiet=getattr(args, "quiet", False), is_debug=getattr(args, "debug", False))
    cli_logger.log_event("SYSTEM", "init", f"run_id={run_id} command=train started", 0, command="train")
    
    dataset_path = Path(args.dataset)
    cli_logger.log_event("TRAIN", "scan", f"Scanning dataset: {dataset_path}", 10, command="train")
    
    from gmdgen.ai.dataset_index import build_dataset_index
    try:
        index = build_dataset_index(dataset_path, force_rebuild=getattr(args, "force_rebuild", False))
        report_file = output_dir / "train_report.json"
        with open(report_file, "w") as f:
            json.dump(index.to_dict(), f, indent=2)
            
        cli_logger.log_event("TRAIN", "complete", f"Training complete. Files indexed: {len(index.documents)}", 100, command="train")
    except Exception as e:
        cli_logger.log_event("ERROR", "train", f"Training failed: {e}", 100, level="ERROR", command="train")
        sys.exit(1)

def cmd_generate(args):
    # Support a test mode where a single-run JSON output is printed for CI tests
    if getattr(args, "test_local_provider", False):
        cfg = {}
        if getattr(args, "config", None):
            try:
                import yaml
                cfg_text = Path(args.config).read_text(encoding="utf-8")
                cfg = yaml.safe_load(cfg_text) or {}
            except Exception:
                cfg = {}
        if getattr(args, "audio_file", None):
            cfg["audio_file"] = str(args.audio_file)
        
        cfg["ai_provider"] = "local_test_only"
        cfg["allow_local_test_provider"] = True
        
        # Call internal generate function and emit JSON to stdout for tests
        result = generate_from_config(cfg)
        print(json.dumps(result))
        return

    # Normal real generation
    run_id, output_dir = get_run_id_and_dir(args.output_dir)
    cli_logger = CLILogger(run_id, output_dir, is_quiet=getattr(args, "quiet", False), is_debug=getattr(args, "debug", False))
    cli_logger.log_event("SYSTEM", "init", f"run_id={run_id} command=generate started", 0, command="generate")
    
    # Seed handling
    seed = args.seed if args.seed is not None else random.randint(0, 999999)
    set_global_seed(seed)
    cli_logger.log_event("CONFIG", "init", f"seed={seed}", 2, command="generate")

    # Validation of API key for Gemini
    api_key_env = args.api_key_env or "GEMINI_API_KEY"
    if not os.environ.get(api_key_env):
        cli_logger.log_event("ERROR", "init", f"API key {api_key_env} is missing", 0, level="ERROR", command="generate")
        sys.exit(1)
    
    cfg = {
        "ai_provider": "gemini",
        "gemini_model": args.model or "gemini-2.5-flash",
        "output_dir": str(output_dir),
        "allow_low_quality_draft": getattr(args, "allow_low_quality_draft", False),
        "seed": seed,
        "repair_level": getattr(args, "repair_level", True),
    }
    
    # Exposing parameters to config
    if args.object_count is not None: cfg["num_objects"] = args.object_count
    if args.min_objects is not None: cfg["min_object_count"] = args.min_objects
    if args.max_objects is not None: cfg["max_object_count"] = args.max_objects
    if args.duration is not None: cfg["target_duration"] = args.duration
    if args.difficulty is not None: cfg["difficulty"] = args.difficulty

    if getattr(args, "config", None):
        try:
            import yaml
            cfg_text = Path(args.config).read_text(encoding="utf-8")
            file_cfg = yaml.safe_load(cfg_text) or {}
            cfg.update(file_cfg)
        except Exception as e:
            cli_logger.log_event("ERROR", "init", f"Failed to load config: {e}", 0, level="ERROR", command="generate")
            sys.exit(1)
            
    if getattr(args, "audio_file", None):
        cfg["audio_file"] = str(args.audio_file)

    cli_logger.log_event("CONFIG", "init", f"provider=gemini model={cfg.get('gemini_model')} fallback=disabled", 5, command="generate")
    
    try:
        if cfg.get("audio_file"):
            cli_logger.log_event("AUDIO", "load", f"Loading audio: {cfg['audio_file']}", 10, command="generate")
            
        cli_logger.log_event("PROVIDER", "call", f"Gemini request started model={cfg.get('gemini_model')}", 40, command="generate")
        
        result = generate_from_config(cfg)
        
        cli_logger.log_event("PROVIDER", "call", "Gemini request complete", 50, command="generate")
        
        final_score = result.get("final_score", 0.0)
        validation = result.get("validation_report", {})
        passed = validation.get("passed", False) if isinstance(validation, dict) else False
        
        is_success = passed and result.get("num_objects", 0) > 0
        cli_logger.log_event("QUALITY_GATE", "check", f"passed={is_success} final_score={final_score:.2f} objects={result.get('num_objects')}", 93, command="generate")
        
        report_file = output_dir / "generation_report.json"
        with open(report_file, "w") as f:
            json.dump(result, f, indent=2)
            
        if not args.dry_run:
            if is_success:
                out_file = output_dir / "generated.gmd"
                with open(out_file, "w") as f:
                    f.write(result.get("output_gmd", "gmd_data"))
                cli_logger.log_event("OUTPUT", "save", f"wrote {out_file}", 100, command="generate")
            else:
                if getattr(args, "allow_low_quality_draft", False):
                    out_file = output_dir / "low_quality_draft.gmd"
                    with open(out_file, "w") as f:
                        f.write(result.get("output_gmd", "gmd_data_draft"))
                    cli_logger.log_event("OUTPUT", "save", f"wrote draft {out_file}", 100, command="generate")
                else:
                    cli_logger.log_event("QUALITY_GATE", "fail", "Quality gate failed, not saving generated.gmd", 100, level="ERROR", command="generate")
                    sys.exit(1)
        else:
            cli_logger.log_event("SYSTEM", "dry-run", "Dry-run complete, skipping file save", 100, command="generate")
    
    except Exception as e:
        cli_logger.log_event("ERROR", "generate", f"Generation failed: {e}", 100, level="ERROR", command="generate")
        
        error_file = output_dir / "error_report.json"
        with open(error_file, "w") as f:
            json.dump({"error": str(e)}, f, indent=2)
        sys.exit(1)


def cmd_validate(args):
    from gmdgen.generate.validator import validate_gmd_file, validate_gmd_text
    run_id, output_dir = get_run_id_and_dir()
    cli_logger = CLILogger(run_id, output_dir, is_quiet=getattr(args, "quiet", False), is_debug=getattr(args, "debug", False))
    cli_logger.log_event("SYSTEM", "init", f"run_id={run_id} command=validate started", 0, command="validate")
    
    try:
        input_path = Path(args.input)
        if input_path.is_file():
            report = validate_gmd_file(input_path)
        else:
            report = validate_gmd_text(args.input)
            
        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
        
        cli_logger.log_event("VALIDATION", "complete", f"Validation complete. Passed: {report.get('passed')}", 100, command="validate")
    except Exception as e:
        cli_logger.log_event("ERROR", "validate", f"Validation failed: {e}", 100, level="ERROR", command="validate")
        sys.exit(1)

def cmd_repair(args):
    from gmdgen.io.gmd_parser import parse_gmd_text
    from gmdgen.io.gmd_writer import serialize_tags
    from gmdgen.generate.repairer import repair_level_objects, repair_report_to_dict
    
    run_id, output_dir = get_run_id_and_dir()
    cli_logger = CLILogger(run_id, output_dir, is_quiet=getattr(args, "quiet", False), is_debug=getattr(args, "debug", False))
    cli_logger.log_event("SYSTEM", "init", f"run_id={run_id} command=repair started", 0, command="repair")
    
    try:
        input_path = Path(args.input)
        text = input_path.read_text(encoding="utf-8")
        tags = parse_gmd_text(text)
        
        level_data = tags.get("k4")
        if not level_data:
            raise ValueError("No k4 level data found in GMD")
            
        from gmdgen.io.gmd_decoder import decode_level_data, encode_level_data
        decoded = decode_level_data(level_data)
        objects = decoded.split(";")
        
        repaired_objects, report = repair_level_objects(objects)
        repaired_text = ";".join(repaired_objects)
        tags["k4"] = encode_level_data(repaired_text)
        
        if args.output:
            out_text = serialize_tags(tags)
            Path(args.output).write_text(out_text, encoding="utf-8")
            
        if args.report:
            with open(args.report, "w") as f:
                json.dump(repair_report_to_dict(report), f, indent=2)
        
        cli_logger.log_event("REPAIR", "complete", f"Repair complete. Fixed issues: {report.total_fixed}", 100, command="repair")
    except Exception as e:
        cli_logger.log_event("ERROR", "repair", f"Repair failed: {e}", 100, level="ERROR", command="repair")
        sys.exit(1)

def cmd_report(args):
    run_id, output_dir = get_run_id_and_dir()
    cli_logger = CLILogger(run_id, output_dir, is_quiet=getattr(args, "quiet", False), is_debug=getattr(args, "debug", False))
    cli_logger.log_event("SYSTEM", "init", f"run_id={run_id} command=report started", 0, command="report")
    
    try:
        with open(args.input, "r") as f:
            data = json.load(f)
            
        # Simplistic textual report for now
        lines = []
        lines.append(f"Level Report: {data.get('output_name', 'unknown')}")
        lines.append(f"Final Score: {data.get('final_score', 0.0)}")
        lines.append(f"Objects: {data.get('num_objects', 0)}")
        
        output_text = "\n".join(lines)
        if args.output:
            Path(args.output).write_text(output_text)
        else:
            print(output_text)
            
        cli_logger.log_event("REPORT", "complete", "Report complete", 100, command="report")
    except Exception as e:
        cli_logger.log_event("ERROR", "report", f"Report failed: {e}", 100, level="ERROR", command="report")
        sys.exit(1)


def main():
    parser = build_parser()
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
        
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "doctor":
        cmd_doctor(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "repair":
        cmd_repair(args)
    elif args.command == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
