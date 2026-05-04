from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gmdgen.ai.cache import ContextCacheRecord, load_context_cache, save_context_cache
from gmdgen.ai.context_index import ContextIndex, build_context_index
from gmdgen.ai.dataset_index import (
    DatasetIndex,
    build_dataset_index,
    ensure_dataset_dir,
    load_dataset_index,
    resolve_dataset_dir,
    save_dataset_index,
)
from gmdgen.utils.config import clone_config_with_updates


@dataclass(slots=True)
class AutoTrainingConfig:
    dataset_dir: str = "dataset"
    context_dirs: list[str] = field(default_factory=lambda: ["docs"])
    reference_level_dirs: list[str] = field(default_factory=lambda: ["tests/fixtures/levels"])
    schema_paths: list[str] = field(default_factory=lambda: ["src/gmdgen/gd/triggers.py", "src/gmdgen/gd/plans.py"])
    cache_dir: str = "outputs/context_cache"
    max_context_chars: int = 12000
    max_total_context_chars: int = 30000
    recursive_dataset_scan: bool = True
    include_all_dataset_files: bool = True
    max_file_size_mb: float = 8.0
    rebuild: bool = False
    use_dataset_cache: bool = True
    use_embeddings_if_available: bool = False
    ollama_model_for_summary: str | None = None


@dataclass(slots=True)
class AutoTrainingResult:
    success: bool
    document_count: int = 0
    reference_level_count: int = 0
    chunk_count: int = 0
    cache_path: str = ""
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    used_cache: bool = False
    trigger_schema_loaded: bool = False
    playability_rules_loaded: bool = False
    rebuild: bool = False
    dataset_dir: str = ""
    dataset_scan_result: dict[str, Any] = field(default_factory=dict)
    dataset_document_count: int = 0
    dataset_chunk_count: int = 0
    dataset_reference_level_count: int = 0
    dataset_failed_files: int = 0
    dataset_skipped_files: int = 0
    cache_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "document_count": self.document_count,
            "reference_level_count": self.reference_level_count,
            "chunk_count": self.chunk_count,
            "cache_path": self.cache_path,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "used_cache": self.used_cache,
            "trigger_schema_loaded": self.trigger_schema_loaded,
            "playability_rules_loaded": self.playability_rules_loaded,
            "rebuild": self.rebuild,
            "dataset_dir": self.dataset_dir,
            "dataset_scan_result": dict(self.dataset_scan_result),
            "dataset_document_count": self.dataset_document_count,
            "dataset_chunk_count": self.dataset_chunk_count,
            "dataset_reference_level_count": self.dataset_reference_level_count,
            "dataset_failed_files": self.dataset_failed_files,
            "dataset_skipped_files": self.dataset_skipped_files,
            "cache_used": self.cache_used,
        }


def run_auto_training(config: AutoTrainingConfig) -> tuple[AutoTrainingResult, ContextIndex]:
    started_at = _now_iso()
    cache_path = Path(config.cache_dir) / "context_index.json"
    dataset_dir = ensure_dataset_dir(config.dataset_dir)
    warnings: list[str] = []
    errors: list[str] = []

    try:
        base_index = build_context_index(
            context_dirs=config.context_dirs,
            reference_level_dirs=config.reference_level_dirs,
            schema_paths=config.schema_paths,
            max_context_chars=config.max_context_chars,
        )
        dataset_index = _load_or_build_dataset_index(config, dataset_dir)
        index = _merge_dataset_index(base_index, dataset_index)
    except Exception as exc:  # noqa: BLE001
        result = AutoTrainingResult(
            success=False,
            cache_path=str(cache_path),
            errors=[f"context_index_build_failed: {exc}"],
            started_at=started_at,
            finished_at=_now_iso(),
            rebuild=bool(config.rebuild),
            dataset_dir=str(dataset_dir),
        )
        return result, ContextIndex()

    if not config.rebuild:
        cached = load_context_cache(cache_path)
        if cached and cached.source_hash == index.source_hash:
            payload = cached.payload
            cached_index = ContextIndex(
                documents=list(payload.get("documents", [])),
                reference_levels=list(payload.get("reference_levels", [])),
                chunks=list(payload.get("chunks", [])),
                source_hash=str(payload.get("source_hash", cached.source_hash)),
            )
            result = AutoTrainingResult(
                success=True,
                document_count=len(base_index.documents),
                reference_level_count=len(base_index.reference_levels),
                chunk_count=len(base_index.chunks),
                cache_path=str(cache_path),
                warnings=warnings,
                errors=errors,
                started_at=started_at,
                finished_at=_now_iso(),
                used_cache=True,
                trigger_schema_loaded=_has_schema(cached_index, "trigger"),
                playability_rules_loaded=_has_schema(cached_index, "playability"),
                rebuild=bool(config.rebuild),
                dataset_dir=str(dataset_dir),
                dataset_scan_result=dict(dataset_index.scan_result),
                dataset_document_count=len(dataset_index.documents),
                dataset_chunk_count=len(dataset_index.chunks),
                dataset_reference_level_count=len(dataset_index.reference_levels),
                dataset_failed_files=int(dataset_index.scan_result.get("files_failed", 0) or 0),
                dataset_skipped_files=int(dataset_index.scan_result.get("files_skipped", 0) or 0),
                cache_used=True,
            )
            return result, cached_index

    save_context_cache(
        cache_path,
        ContextCacheRecord(
            source_hash=index.source_hash,
            updated_at=_now_iso(),
            payload=index.to_dict(),
        ),
    )
    result = AutoTrainingResult(
        success=True,
        document_count=len(base_index.documents),
        reference_level_count=len(base_index.reference_levels),
        chunk_count=len(base_index.chunks),
        cache_path=str(cache_path),
        warnings=warnings,
        errors=errors,
        started_at=started_at,
        finished_at=_now_iso(),
        used_cache=False,
        trigger_schema_loaded=_has_schema(index, "trigger"),
        playability_rules_loaded=_has_schema(index, "playability"),
        rebuild=bool(config.rebuild),
        dataset_dir=str(dataset_dir),
        dataset_scan_result=dict(dataset_index.scan_result),
        dataset_document_count=len(dataset_index.documents),
        dataset_chunk_count=len(dataset_index.chunks),
        dataset_reference_level_count=len(dataset_index.reference_levels),
        dataset_failed_files=int(dataset_index.scan_result.get("files_failed", 0) or 0),
        dataset_skipped_files=int(dataset_index.scan_result.get("files_skipped", 0) or 0),
        cache_used=False,
    )
    return result, index


def rebuild_auto_training_config(config: AutoTrainingConfig, *, rebuild: bool = True) -> AutoTrainingConfig:
    cloned = clone_config_with_updates(config, rebuild=rebuild)
    if isinstance(cloned, AutoTrainingConfig):
        return cloned
    return AutoTrainingConfig(**cloned)


def invalidate_cache_if_sources_changed(config: AutoTrainingConfig) -> bool:
    cache_path = Path(config.cache_dir) / "context_index.json"
    cached = load_context_cache(cache_path)
    if cached is None:
        return True
    dataset_dir = resolve_dataset_dir(config.dataset_dir)
    dataset_index = _load_or_build_dataset_index(config, dataset_dir)
    fresh_index = build_context_index(
        context_dirs=config.context_dirs,
        reference_level_dirs=config.reference_level_dirs,
        schema_paths=config.schema_paths,
        max_context_chars=config.max_context_chars,
    )
    fresh_index = _merge_dataset_index(fresh_index, dataset_index)
    return fresh_index.source_hash != cached.source_hash


def _load_or_build_dataset_index(config: AutoTrainingConfig, dataset_dir: Path) -> DatasetIndex:
    cache_path = dataset_dir / "cache" / "dataset_index.json"
    if config.use_dataset_cache and not config.rebuild:
        cached = load_dataset_index(cache_path)
        if cached is not None:
            return cached
    built = build_dataset_index(
        dataset_dir,
        recursive=bool(config.recursive_dataset_scan),
        max_file_size_mb=float(config.max_file_size_mb),
        max_total_context_chars=int(config.max_total_context_chars),
    )
    if config.use_dataset_cache:
        save_dataset_index(built, cache_path)
    return built


def _merge_dataset_index(index: ContextIndex, dataset_index: DatasetIndex) -> ContextIndex:
    documents = list(index.documents)
    reference_levels = list(index.reference_levels)
    chunks = list(index.chunks)
    documents.extend(
        {
            "path": f"dataset/{doc.get('relative_path', doc.get('path', ''))}",
            "title": doc.get("file_name", doc.get("relative_path", "dataset")),
            "kind": doc.get("document_type", "dataset"),
        }
        for doc in dataset_index.documents
    )
    reference_levels.extend(dataset_index.reference_levels)
    chunks.extend(dataset_index.chunks)
    joined_hash = f"{index.source_hash}:{dataset_index.dataset_hash}".encode("utf-8", errors="replace")
    import hashlib

    return ContextIndex(
        documents=documents,
        reference_levels=reference_levels,
        chunks=chunks[: max(1, len(chunks))],
        source_hash=hashlib.sha256(joined_hash).hexdigest(),
    )


def _has_schema(index: ContextIndex, token: str) -> bool:
    lowered = token.lower()
    for doc in index.documents:
        path = str(doc.get("path", "")).lower()
        title = str(doc.get("title", "")).lower()
        if lowered in path or lowered in title:
            return True
    return False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
