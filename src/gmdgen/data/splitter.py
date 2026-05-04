# SPDX-License-Identifier: GPL-3.0-or-later
"""Hold-out split utilities for GD level datasets.

Goodfellow Ch.5 §5.3 "Hyperparameters and Validation Sets":
  'The training set is used to learn the parameters. The validation set is
  used to estimate the generalization error during or after training, in
  order to update hyperparameters.'

  We don't have hyperparameters in the gradient-descent sense, but we need
  a held-out set to verify that the generation pipeline produces files that:
    - parse correctly when re-imported
    - have acceptable x_monotone_ratio
    - have reasonable section count and density distribution

Split strategy:
  - Deterministic (seed-based) so splits are reproducible.
  - Stratified by object_count bucket so the validation set covers
    both short and long levels.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from gmdgen.data.loader import load_dataset_with_report
from gmdgen.data.preprocess import split_level_objects
from gmdgen.data.schema import DatasetLoadResult, GMDRecord


@dataclass
class DatasetSplit:
    train: list[GMDRecord]
    validation: list[GMDRecord]

    @property
    def train_count(self) -> int:
        return len(self.train)

    @property
    def val_count(self) -> int:
        return len(self.validation)


def _object_count_bucket(record: GMDRecord, bins: list[int]) -> int:
    count = len(split_level_objects(record.decoded_level_data))
    for i, threshold in enumerate(bins):
        if count <= threshold:
            return i
    return len(bins)


def split_dataset(
    records: list[GMDRecord],
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
    min_val_records: int = 1,
) -> DatasetSplit:
    """Split records into train/validation with stratification by object count.

    Parameters
    ----------
    records:         all loaded GMDRecord instances
    val_ratio:       fraction to use for validation (default 0.2 = 20%)
    seed:            RNG seed for reproducibility
    min_val_records: at least this many records in validation set
    """
    if not records:
        return DatasetSplit(train=[], validation=[])

    rng = random.Random(seed)

    # Stratify by object count quintile
    size_bins = [100, 500, 2000, 10000, 50000]
    buckets: dict[int, list[GMDRecord]] = {}
    for record in records:
        b = _object_count_bucket(record, size_bins)
        buckets.setdefault(b, []).append(record)

    train_records: list[GMDRecord] = []
    val_records: list[GMDRecord] = []

    for bucket_records in buckets.values():
        shuffled = list(bucket_records)
        rng.shuffle(shuffled)
        n_val = max(min_val_records, round(len(shuffled) * val_ratio))
        n_val = min(n_val, len(shuffled))
        val_records.extend(shuffled[:n_val])
        train_records.extend(shuffled[n_val:])

    # Guarantee at least min_val_records in validation
    if len(val_records) < min_val_records and len(train_records) > 0:
        needed = min_val_records - len(val_records)
        val_records.extend(train_records[:needed])
        train_records = train_records[needed:]

    return DatasetSplit(train=train_records, validation=val_records)


def split_dataset_dir(
    dataset_dir: Path,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
    min_val_records: int = 1,
) -> tuple[DatasetSplit, DatasetLoadResult]:
    """Load and split a dataset directory in one step."""
    load_result = load_dataset_with_report(dataset_dir)
    split = split_dataset(
        load_result.records,
        val_ratio=val_ratio,
        seed=seed,
        min_val_records=min_val_records,
    )
    return split, load_result
