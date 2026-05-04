from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from gmdgen.train.holdout_validator import run_holdout_validation
from gmdgen.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hold-out validation on the dataset.")
    parser.add_argument(
        "--dataset",
        default="dataset",
        help="Path to the .gmd dataset directory",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of records used as validation (default 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible split (default 42)",
    )
    parser.add_argument(
        "--min-val",
        type=int,
        default=1,
        help="Minimum validation records (default 1)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)

    summary = run_holdout_validation(
        Path(args.dataset),
        val_ratio=args.val_ratio,
        seed=args.seed,
        min_val_records=args.min_val,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
