# SPDX-License-Identifier: GPL-3.0-or-later
import sys
import os
import argparse
import logging
import time
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

from gmdgen.ai.factory import create_ai_provider_from_config
from gmdgen.audio.analysis import analyze_audio
from gmdgen.ai.context_index import build_context_index
from gmdgen.generate.generator import generate_from_config

# Global logger setup
logger = logging.getLogger("gmdgen.cli")

class CLILogger:
    def __init__(self, run_id: str, output_dir: Path):
        self.run_id = run_id
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "run.log"
        self.events_file = self.output_dir / "events.jsonl"
        
        # Setup file handler
        fh = logging.FileHandler(self.log_file)
        fh.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'))
        logging.getLogger().addHandler(fh)

    def log_event(self, category: str, stage: str, message: str, progress: int = 0, level: str = "INFO"):
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "run_id": self.run_id,
            "category": category,
            "stage": stage,
            "level": level,
            "progress_percent": progress,
            "message": message
        }
        with open(self.events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        
        # Also print to console
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [{category}] [{progress}%] {message}")

def cmd_doctor(args):
    print("=== gmdgen doctor ===")
    print(f"Python: {sys.version}")
    print(f"Root: {Path.cwd()}")
    
    # Check Gemini API Key
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        print("[OK] GEMINI_API_KEY found")
    else:
        print("[ERR] GEMINI_API_KEY missing. Set it with export GEMINI_API_KEY='your-key'")
    
    # Check dependencies
    try:
        from google import genai
        print("[OK] google-genai installed")
    except ImportError:
        print("[ERR] google-genai missing. Run 'pip install google-genai'")

def cmd_train(args):
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    output_dir = Path("outputs/runs") / run_id
    cli_logger = CLILogger(run_id, output_dir)
    
    cli_logger.log_event("SYSTEM", "init", f"Starting train command, run_id={run_id}", 0)
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        cli_logger.log_event("ERROR", "init", f"Dataset path {dataset_path} not found", 0, "ERROR")
        return

    cli_logger.log_event("TRAIN", "ingestion", f"Scanning dataset: {dataset_path}", 20)
    
    try:
        index = build_context_index(
            context_dirs=[dataset_path],
            reference_level_dirs=[dataset_path],
            max_context_chars=12000
        )
        
        index_file = output_dir / "dataset_index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(index.to_dict(), ensure_ascii=False, indent=2))
            
        cli_logger.log_event("TRAIN", "complete", f"Index saved to {index_file}", 100)
    except Exception as e:
        cli_logger.log_event("ERROR", "train", f"Training failed: {e}", 0, "ERROR")

def cmd_generate(args):
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    output_dir = Path("outputs/runs") / run_id
    cli_logger = CLILogger(run_id, output_dir)
    
    cli_logger.log_event("SYSTEM", "init", f"Starting generate command, run_id={run_id}", 0)
    cli_logger.log_event("CONFIG", "init", f"Provider: {args.provider}, Model: {args.model or 'gemini-2.5-flash'}", 5)
    
    config = {
        "ai_provider": args.provider,
        "gemini_model": args.model or "gemini-2.5-flash",
        "audio_file": args.audio,
        "output_dir": str(output_dir),
        "run_id": run_id
    }
    
    try:
        cli_logger.log_event("AUDIO", "analysis", f"Analyzing audio: {args.audio}", 20)
        # Actual audio analysis would be called inside generate_from_config or here
        
        cli_logger.log_event("PROVIDER", "call", f"Requesting {args.provider} API...", 40)
        
        # This calls the actual generation pipeline
        result = generate_from_config(config)
        
        if result.get("success"):
            cli_logger.log_event("OUTPUT", "save", f"Level saved to {result.get('output_path')}", 100)
        else:
            cli_logger.log_event("QUALITY_GATE", "failed", "Quality gate failed, saving draft only", 90, "WARNING")
            
    except Exception as e:
        cli_logger.log_event("ERROR", "generate", f"Generation failed: {e}", 0, "ERROR")

def main():
    parser = argparse.ArgumentParser(description="gmdgen: Geometry Dash AI Level Generator (Gemini-first CLI)")
    subparsers = parser.add_subparsers(dest="command")

    # Doctor
    subparsers.add_parser("doctor", help="Check environment and configuration")

    # Train
    train_p = subparsers.add_parser("train", help="Build dataset context")
    train_p.add_argument("--dataset", default="dataset", help="Path to dataset directory")

    # Generate
    gen_p = subparsers.add_parser("generate", help="Generate a new level")
    gen_p.add_argument("--provider", default="gemini", choices=["gemini", "ollama"], help="AI provider")
    gen_p.add_argument("--model", help="Model name (e.g., gemini-2.5-flash)")
    gen_p.add_argument("--audio", required=True, help="Path to audio file")
    gen_p.add_argument("--output", default="outputs", help="Output directory")

    args = parser.parse_args()

    if args.command == "doctor":
        cmd_doctor(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "generate":
        cmd_generate(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
