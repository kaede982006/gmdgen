from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def export_jsonl_for_finetuning(dataset_dir: Path, output_path: Path) -> None:
    # Read ML records and export to JSONL
    # - filters low quality outputs
    # - checks quality_gate_passed=True
    # - checks user_rating >= threshold
    # - checks repair_loss <= max
    pass


def export_preference_dataset(dataset_dir: Path, output_path: Path) -> None:
    # Exports pairs of chosen / rejected candidates
    pass


def export_reward_model_dataset(dataset_dir: Path, output_path: Path) -> None:
    pass
