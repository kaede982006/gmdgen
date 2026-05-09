# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path
from typing import Any

from gmdgen.io.gmd_parser import parse_gmd_file

def evaluate_reference_levels(dataset_dir: str | Path) -> dict[str, Any]:
    base = Path(dataset_dir)
    refs_dir = base / "reference_levels"
    if not refs_dir.exists():
        return {"error": "reference_levels directory does not exist"}

    valid_files = []
    invalid_files = []
    motifs_estimated = 0
    tags_found = set()
    hashes = set()
    duplicates = []

    for gmd_file in refs_dir.rglob("*.gmd"):
        try:
            level = parse_gmd_file(gmd_file)
            content_hash = hash(level.raw_text)
            if content_hash in hashes:
                duplicates.append(str(gmd_file))
                continue
            hashes.add(content_hash)
            
            # Simple motif estimation based on object count
            motifs_estimated += max(1, len(level.raw_text) // 500)
            valid_files.append(str(gmd_file))
            
            # Simulated tag extraction
            if "demon" in str(gmd_file).lower():
                tags_found.add("demon")
            if "layout" in str(gmd_file).lower():
                tags_found.add("layout")
        except Exception as e:
            invalid_files.append({"file": str(gmd_file), "error": str(e)})

    return {
        "valid_count": len(valid_files),
        "invalid_count": len(invalid_files),
        "duplicate_count": len(duplicates),
        "motifs_estimated": motifs_estimated,
        "tags_detected": list(tags_found),
        "invalid_files": invalid_files,
        "duplicates": duplicates
    }
