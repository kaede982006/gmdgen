# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


import yaml

from gmdgen.gui.app import launch_gui
from gmdgen.generate.generator import generate_from_config
from gmdgen.train.holdout_validator import run_holdout_validation
from gmdgen.train.trainer import train_from_config
from gmdgen.utils.logging import configure_logging


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"Config must be a mapping: {path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gmdgen",
        description="GUI-first Geometry Dash generator (external AI API-only for real generation).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=False)

    subparsers.add_parser("gui", help="Launch GUI application")

    train_parser = subparsers.add_parser("train", help="Train a model artifact")
    train_parser.add_argument(
        "--config",
        default="configs/train.yaml",
        help="Path to train config yaml",
    )

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate a new gmd file from artifact",
    )
    generate_parser.add_argument(
        "--config",
        default="configs/generate.yaml",
        help="Path to generate config yaml",
    )
    generate_parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt text to bias generation style.",
    )
    generate_parser.add_argument(
        "--audio-file",
        "--audio",
        dest="audio_file",
        default=None,
        help="Path to an audio file for audio-conditioned Geometry Dash level generation.",
    )
    generate_parser.add_argument(
        "--ai-provider",
        choices=("ollama", "ollama"),
        default=None,
        help="AI planning provider. Real generation requires an external AI API.",
    )
    generate_parser.add_argument(
        "--test-local-provider",
        action="store_true",
        help="Development smoke-test mode only: use local mock provider and mark output as test-only.",
    )
    generate_parser.add_argument(
        "--output",
        default=None,
        help="Output .gmd path. Sets output_dir and output_name.",
    )
    generate_parser.add_argument("--ollama-model", default=None, help="Ollama model name.")
    generate_parser.add_argument("--low-cost-mode", action="store_true", help="Use compact low-cost AI planning.")
    generate_parser.add_argument("--max-ai-calls", type=int, default=None, help="Maximum external AI calls per generation.")
    generate_parser.add_argument("--ollama-context-dir", default=None, help="Directory with local context documents.")
    generate_parser.add_argument(
        "--ollama-reference-levels-dir",
        default=None,
        help="Directory with reference .gmd levels to summarize for planning.",
    )
    generate_parser.add_argument(
        "--ollama-enable-retrieval",
        action="store_true",
        help="Use local keyword retrieval over ollama-context-dir.",
    )
    generate_parser.add_argument("--ollama-debug", action="store_true", help="Enable Ollama planning debug metadata.")
    generate_parser.add_argument(
        "--ollama-save-debug-artifacts",
        action="store_true",
        help="Write sanitized Ollama planning debug artifacts.",
    )
    generate_parser.add_argument("--ollama-debug-dir", default=None, help="Directory for Ollama debug artifacts.")
    generate_parser.add_argument(
        "--no-ollama-fallback",
        action="store_true",
        help="Raise on Ollama planning failure instead of using local fallback.",
    )
    generate_parser.add_argument(
        "--export-finetune-jsonl",
        default=None,
        help="Append/export a structured generation example JSONL for future fine-tuning.",
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Run hold-out validation on the dataset",
    )
    validate_parser.add_argument(
        "--dataset",
        default="dataset",
        help="Path to the .gmd dataset directory (default: dataset)",
    )
    validate_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of records used as validation (default 0.2)",
    )
    validate_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible split (default 42)",
    )
    validate_parser.add_argument(
        "--min-val",
        type=int,
        default=1,
        help="Minimum validation records (default 1)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(verbose=bool(args.verbose))

    if args.command in {None, "gui"}:
        raise SystemExit(launch_gui())

    if args.command == "train":
        config = _load_yaml(Path(args.config))
        result = train_from_config(config)

    elif args.command == "generate":
        if not args.test_local_provider:
            parser.exit(
                2,
                "error: This program is GUI-only for user generation. "
                "Run `python -m gmdgen` or `python -m gmdgen gui`.\n",
            )
        config = _load_yaml(Path(args.config))
        if args.prompt is not None:
            config["prompt"] = args.prompt
        if args.audio_file is not None:
            config["audio_file"] = args.audio_file
        if args.output is not None:
            output_path = Path(args.output)
            config["output_dir"] = str(output_path.parent if str(output_path.parent) else Path("."))
            config["output_name"] = output_path.stem
        for attr in (
            "ai_provider",
            "ollama_model",
            "ollama_model",
            "ollama_context_dir",
            "ollama_reference_levels_dir",
            "ollama_debug_dir",
            "export_finetune_jsonl",
        ):
            value = getattr(args, attr, None)
            if value is not None:
                config[attr] = value
        if args.ollama_enable_retrieval:
            config["ollama_enable_retrieval"] = True
        if args.low_cost_mode:
            config["low_cost_mode"] = True
        if args.max_ai_calls is not None:
            config["max_ai_calls_per_generation"] = args.max_ai_calls
        if args.ollama_debug:
            config["ollama_debug"] = True
        if args.ollama_save_debug_artifacts:
            config["ollama_save_debug_artifacts"] = True
        if args.test_local_provider:
            config["allow_local_test_provider"] = True
            config["ai_provider"] = "local_test_only"
            config["generation_mode"] = "test_local_mock"
        if args.no_ollama_fallback:
            config["ollama_fallback_to_local"] = False
        try:
            result = generate_from_config(config)
        except Exception as exc:  # noqa: BLE001
            parser.exit(1, f"error: {exc}\n")

    elif args.command == "validate":
        result = run_holdout_validation(
            Path(args.dataset),
            val_ratio=args.val_ratio,
            seed=args.seed,
            min_val_records=args.min_val,
        )

    else:
        raise ValueError(f"Unknown command: {args.command}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
