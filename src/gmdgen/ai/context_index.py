from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gmdgen.ai.context import (
    load_context_documents,
    load_reference_levels,
    summarize_context_documents,
)


@dataclass(slots=True)
class ContextIndex:
    documents: list[dict[str, Any]] = field(default_factory=list)
    reference_levels: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    source_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "reference_levels": self.reference_levels,
            "chunks": self.chunks,
            "source_hash": self.source_hash,
        }


def build_context_index(
    *,
    context_dirs: list[str | Path],
    reference_level_dirs: list[str | Path],
    schema_paths: list[str | Path] | None = None,
    max_context_chars: int = 12000,
) -> ContextIndex:
    all_docs = []
    all_refs = []
    schema_paths = schema_paths or []
    for path in context_dirs:
        all_docs.extend(load_context_documents(path))
    for path in reference_level_dirs:
        all_refs.extend(load_reference_levels(path))
    for schema_path in schema_paths:
        schema = Path(schema_path)
        if schema.exists() and schema.is_file():
            all_docs.append(
                type("Doc", (), {  # simple shape adapter
                    "path": str(schema),
                    "title": schema.name,
                    "text": schema.read_text(encoding="utf-8", errors="replace"),
                    "kind": schema.suffix.lstrip("."),
                })()
            )
    chunks = summarize_context_documents(all_docs, max_context_chars)
    source_hash = compute_source_hash(context_dirs, reference_level_dirs, schema_paths)
    return ContextIndex(
        documents=[
            {"path": getattr(doc, "path", ""), "title": getattr(doc, "title", ""), "kind": getattr(doc, "kind", "")}
            for doc in all_docs
        ],
        reference_levels=all_refs,
        chunks=chunks,
        source_hash=source_hash,
    )


def compute_source_hash(
    context_dirs: list[str | Path],
    reference_level_dirs: list[str | Path],
    schema_paths: list[str | Path] | None = None,
) -> str:
    schema_paths = schema_paths or []
    fingerprints: list[str] = []
    for root in list(context_dirs) + list(reference_level_dirs):
        path = Path(root)
        if not path.exists():
            continue
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file():
                stat = file_path.stat()
                fingerprints.append(f"{file_path}:{stat.st_mtime_ns}:{stat.st_size}")
    for schema in schema_paths:
        path = Path(schema)
        if path.exists() and path.is_file():
            stat = path.stat()
            fingerprints.append(f"{path}:{stat.st_mtime_ns}:{stat.st_size}")
    joined = "\n".join(fingerprints).encode("utf-8", errors="replace")
    return hashlib.sha256(joined).hexdigest()


def context_index_to_json(index: ContextIndex) -> str:
    return json.dumps(index.to_dict(), ensure_ascii=False, sort_keys=True)
