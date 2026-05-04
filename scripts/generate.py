from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from gmdgen.generate.generator import generate_from_config
from gmdgen.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate gmd from trained model.")
    parser.add_argument("--config", default="configs/generate.yaml")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt text to bias generation style (e.g. 'fast wave hell').",
    )
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    if args.prompt is not None:
        config["prompt"] = args.prompt

    result = generate_from_config(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
