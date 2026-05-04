from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from gmdgen.train.trainer import train_from_config
from gmdgen.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Train gmdgen model.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}

    result = train_from_config(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
