import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

@dataclass
class DatasetManifest:
    dataset_version: str = "1.0"
    created_at: str = ""
    total_reference_levels: int = 0
    total_learning_examples: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    rejected_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def initialize_ml_dataset_structure(base_dir: Path) -> None:
    for folder in ["features", "embeddings", "labels", "train", "val", "test", "rejected", "manifests"]:
        (base_dir / "ml" / folder).mkdir(parents=True, exist_ok=True)

def split_train_val_test(examples: list[dict[str, Any]], train_ratio: float = 0.8, val_ratio: float = 0.1) -> dict[str, list[dict[str, Any]]]:
    random.seed(42)  # Deterministic split
    valid_examples = [e for e in examples if e.get("status") not in {"rejected", "low_quality"}]
    rejected_examples = [e for e in examples if e.get("status") in {"rejected", "low_quality"}]
    
    shuffled = list(valid_examples)
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
        "rejected": rejected_examples
    }

def export_preference_pairs(examples: list[dict[str, Any]], output_path: Path) -> None:
    pairs = []
    # Simplified mock for preference pairs export
    for i in range(len(examples) - 1):
        if examples[i].get("user_rating", 0) > examples[i+1].get("user_rating", 0):
            pairs.append({"chosen": examples[i], "rejected": examples[i+1]})
    
    output_path.write_text(json.dumps(pairs, indent=2), encoding="utf-8")
