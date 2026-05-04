from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gmdgen.ai.context import summarize_reference_level
from gmdgen.learning.feature_extractor import learn_from_file
from gmdgen.learning.store import sanitize_learning_payload


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
SUPPORTED_DATASET_EXTENSIONS = {
    ".md",
    ".txt",
    ".json",
    ".jsonl",
    ".py",
    ".gmd",
    ".csv",
    ".yaml",
    ".yml",
}
SKIPPED_DIR_NAMES = {"__pycache__", ".pytest_cache", ".git", "cache", "debug", "exports", "rejected"}


@dataclass(slots=True)
class DatasetPathStatus:
    path: str
    exists: bool
    is_dir: bool
    readable: bool
    is_empty: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DatasetScanResult:
    dataset_dir: str
    total_files_found: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    total_bytes: int = 0
    indexed_extensions: dict[str, int] = field(default_factory=dict)
    skipped_extensions: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DatasetDocument:
    path: str
    relative_path: str
    file_name: str
    extension: str
    size_bytes: int
    modified_time: str
    content_hash: str
    document_type: str
    text_summary: str
    chunks: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DatasetIndex:
    documents: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    reference_levels: list[dict[str, Any]] = field(default_factory=list)
    trigger_schema_docs: list[dict[str, Any]] = field(default_factory=list)
    gd_logic_docs: list[dict[str, Any]] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    dataset_hash: str = ""
    cache_path: str = ""
    scan_result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DatasetContextForGeneration:
    relevant_docs_summary: str = ""
    relevant_motifs: list[dict[str, Any]] = field(default_factory=list)
    trigger_schema_summary: str = ""
    gd_logic_summary: str = ""
    style_memory_summary: dict[str, Any] = field(default_factory=dict)
    failure_patterns_summary: list[str] = field(default_factory=list)
    success_patterns_summary: list[str] = field(default_factory=list)
    dataset_stats_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_dataset_dir(path: str | Path | None = None, *, project_root: str | Path | None = None) -> Path:
    root = Path(project_root).expanduser() if project_root else PROJECT_ROOT
    selected: str | Path | None = path
    if selected is None or str(selected).strip() == "":
        selected = os.environ.get("GMDGEN_DATASET_DIR") or "dataset"
    resolved = Path(selected).expanduser()
    if not resolved.is_absolute():
        resolved = root / resolved
    return resolved.resolve()


def ensure_dataset_dir(path: str | Path | None = None, *, project_root: str | Path | None = None) -> Path:
    resolved = resolve_dataset_dir(path, project_root=project_root)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def validate_dataset_dir(path: str | Path | None = None, *, project_root: str | Path | None = None) -> DatasetPathStatus:
    resolved = resolve_dataset_dir(path, project_root=project_root)
    warnings: list[str] = []
    errors: list[str] = []
    exists = resolved.exists()
    is_dir = resolved.is_dir()
    readable = False
    is_empty = True
    if exists and is_dir:
        try:
            children = list(resolved.iterdir())
            readable = True
            is_empty = len(children) == 0
            if is_empty:
                warnings.append("dataset directory is empty")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"dataset directory is not readable: {exc}")
    elif exists and not is_dir:
        errors.append("dataset path exists but is not a directory")
    else:
        warnings.append("dataset directory does not exist yet")
    return DatasetPathStatus(
        path=str(resolved),
        exists=exists,
        is_dir=is_dir,
        readable=readable,
        is_empty=is_empty,
        warnings=warnings,
        errors=errors,
    )


def dataset_cache_path(dataset_dir: str | Path | None = None) -> Path:
    return resolve_dataset_dir(dataset_dir) / "cache" / "dataset_index.json"


def scan_dataset_dir(
    dataset_dir: str | Path | None = None,
    *,
    recursive: bool = True,
    max_file_size_mb: float = 8.0,
) -> DatasetScanResult:
    root = ensure_dataset_dir(dataset_dir)
    result = DatasetScanResult(dataset_dir=str(root))
    iterator = root.rglob("*") if recursive else root.glob("*")
    max_bytes = int(max(0.1, float(max_file_size_mb)) * 1024 * 1024)
    for path in sorted(iterator):
        if not path.is_file():
            continue
        if _should_skip_path(path, root):
            continue
        result.total_files_found += 1
        suffix = path.suffix.lower()
        size = _safe_file_size(path)
        result.total_bytes += size
        if suffix == ".zip":
            result.files_skipped += 1
            _inc(result.skipped_extensions, suffix)
            result.warnings.append(f"zip unsupported skipped: {_relative(path, root)}")
            continue
        if suffix not in SUPPORTED_DATASET_EXTENSIONS:
            result.files_skipped += 1
            _inc(result.skipped_extensions, suffix or "<none>")
            result.warnings.append(f"unsupported extension skipped: {_relative(path, root)}")
            continue
        if size > max_bytes:
            result.files_skipped += 1
            _inc(result.skipped_extensions, suffix)
            result.warnings.append(f"file too large skipped: {_relative(path, root)}")
            continue
        result.files_indexed += 1
        _inc(result.indexed_extensions, suffix)
    return result


def load_dataset_document(
    path: str | Path,
    *,
    dataset_dir: str | Path | None = None,
    max_chars: int = 12000,
) -> DatasetDocument:
    file_path = Path(path).expanduser()
    root = resolve_dataset_dir(dataset_dir) if dataset_dir is not None else file_path.parent
    stat = file_path.stat()
    raw_text, warnings = _read_supported_document(file_path, max_chars=max_chars)
    document_type = classify_dataset_document(file_path, raw_text)
    summary = _summarize_document_text(file_path, raw_text, document_type=document_type, max_chars=max_chars)
    content_hash = hashlib.sha256(raw_text.encode("utf-8", errors="replace")).hexdigest()
    relative = _relative(file_path, root)
    document = DatasetDocument(
        path=relative,
        relative_path=relative,
        file_name=file_path.name,
        extension=file_path.suffix.lower(),
        size_bytes=int(stat.st_size),
        modified_time=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        content_hash=content_hash,
        document_type=document_type,
        text_summary=summary,
        warnings=warnings,
    )
    document.chunks = chunk_dataset_document(document)
    return document


def classify_dataset_document(path: str | Path, content: str = "") -> str:
    suffix = Path(path).suffix.lower()
    lowered = (Path(path).name + "\n" + content[:2000]).lower()
    if "debug" in lowered and "artifact" in lowered:
        return "debug_artifact"
    if "rejected" in lowered or "low_quality" in lowered:
        return "rejected_example"
    if suffix == ".gmd":
        return "reference_level"
    if suffix in {".json", ".jsonl"} and ("validation_report" in lowered or "candidate" in lowered):
        return "generation_example"
    if "trigger" in lowered and "schema" in lowered:
        return "trigger_schema"
    if "time-x" in lowered or "geode" in lowered or "leveltools" in lowered:
        return "gd_logic"
    if suffix in {".md", ".txt", ".py", ".yaml", ".yml", ".csv"}:
        return "context_doc"
    return "dataset_doc"


def chunk_dataset_document(document: DatasetDocument, *, max_chars: int = 1800) -> list[dict[str, Any]]:
    text = document.text_summary or ""
    chunks: list[dict[str, Any]] = []
    for idx, start in enumerate(range(0, len(text), max_chars)):
        chunk = text[start:start + max_chars]
        if not chunk:
            continue
        chunks.append(
            {
                "source": document.relative_path,
                "path": document.relative_path,
                "title": document.file_name,
                "kind": document.document_type,
                "text": chunk,
                "chunk_index": idx,
            }
        )
        if idx >= 5:
            break
    return chunks


def build_dataset_index(
    dataset_dir: str | Path | None = None,
    *,
    recursive: bool = True,
    max_file_size_mb: float = 8.0,
    max_total_context_chars: int = 30000,
) -> DatasetIndex:
    root = ensure_dataset_dir(dataset_dir)
    scan = scan_dataset_dir(root, recursive=recursive, max_file_size_mb=max_file_size_mb)
    documents: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    reference_levels: list[dict[str, Any]] = []
    trigger_schema_docs: list[dict[str, Any]] = []
    gd_logic_docs: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    errors = list(scan.errors)
    warnings = list(scan.warnings)
    remaining = max(0, int(max_total_context_chars))
    for file_path in _iter_supported_files(root, recursive=recursive, max_file_size_mb=max_file_size_mb):
        try:
            document = load_dataset_document(file_path, dataset_dir=root)
        except Exception as exc:  # noqa: BLE001
            scan.files_failed += 1
            errors.append(f"load_failed:{_relative(file_path, root)}:{exc}")
            continue
        if document.document_type in {"debug_artifact", "rejected_example"}:
            scan.files_skipped += 1
            continue
        documents.append(_document_metadata(document))
        if document.document_type == "reference_level":
            reference_levels.append(_reference_summary(file_path, root))
        elif document.document_type == "trigger_schema":
            trigger_schema_docs.append(_document_metadata(document))
        elif document.document_type == "gd_logic":
            gd_logic_docs.append(_document_metadata(document))
        elif document.document_type == "generation_example":
            examples.append(_example_summary(file_path, document))
        for chunk in document.chunks:
            if remaining <= 0:
                break
            text = str(chunk.get("text", ""))[:remaining]
            copied = dict(chunk)
            copied["text"] = text
            chunks.append(copied)
            remaining -= len(text)
    scan.errors = errors
    scan.warnings = warnings
    dataset_hash = _dataset_hash(root, documents)
    cache_path = dataset_cache_path(root)
    return DatasetIndex(
        documents=documents,
        chunks=chunks,
        reference_levels=reference_levels,
        trigger_schema_docs=trigger_schema_docs,
        gd_logic_docs=gd_logic_docs,
        examples=examples,
        created_at=_now_iso(),
        dataset_hash=dataset_hash,
        cache_path=str(cache_path),
        scan_result=scan.to_dict(),
    )


def save_dataset_index(index: DatasetIndex | dict[str, Any], cache_path: str | Path | None = None) -> Path:
    payload = index.to_dict() if isinstance(index, DatasetIndex) else dict(index)
    path = Path(cache_path or payload.get("cache_path") or dataset_cache_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_learning_payload(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_dataset_index(cache_path: str | Path | None = None) -> DatasetIndex | None:
    path = Path(cache_path or dataset_cache_path())
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None
    return DatasetIndex(
        documents=list(payload.get("documents", [])),
        chunks=list(payload.get("chunks", [])),
        reference_levels=list(payload.get("reference_levels", [])),
        trigger_schema_docs=list(payload.get("trigger_schema_docs", [])),
        gd_logic_docs=list(payload.get("gd_logic_docs", [])),
        examples=list(payload.get("examples", [])),
        created_at=str(payload.get("created_at", "")),
        dataset_hash=str(payload.get("dataset_hash", "")),
        cache_path=str(payload.get("cache_path", str(path))),
        scan_result=dict(payload.get("scan_result", {})),
    )


def invalidate_dataset_cache_if_changed(dataset_dir: str | Path | None = None, cache_path: str | Path | None = None) -> bool:
    root = ensure_dataset_dir(dataset_dir)
    cached = load_dataset_index(cache_path or dataset_cache_path(root))
    if cached is None:
        return True
    current = build_dataset_index(root, max_total_context_chars=4000)
    return current.dataset_hash != cached.dataset_hash


def retrieve_relevant_dataset_chunks(index: DatasetIndex, query: str, *, top_k: int = 8) -> list[dict[str, Any]]:
    terms = {term.lower() for term in str(query).replace("_", " ").split() if term.strip()}
    scored: list[tuple[float, dict[str, Any]]] = []
    for chunk in index.chunks:
        text = json.dumps(chunk, ensure_ascii=False).lower()
        score = sum(text.count(term) for term in terms) if terms else 0
        if "reference_level" in text:
            score += 1
        if score > 0 or not terms:
            scored.append((float(score), chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [dict(chunk) for _score, chunk in scored[:top_k]]


def summarize_dataset_context_for_prompt(chunks: list[dict[str, Any]], *, max_chars: int = 4000) -> str:
    remaining = max(0, int(max_chars))
    parts: list[str] = []
    for chunk in chunks:
        if remaining <= 0:
            break
        title = str(chunk.get("title") or chunk.get("source") or "dataset")
        text = str(chunk.get("text", ""))
        item = f"[{title}] {text}"
        clipped = item[:remaining]
        parts.append(clipped)
        remaining -= len(clipped)
    return "\n\n".join(parts)


def build_generation_context_from_dataset(
    dataset_index: DatasetIndex,
    generation_request: dict[str, Any] | None = None,
    *,
    max_chars: int = 5000,
) -> DatasetContextForGeneration:
    request = generation_request or {}
    query = " ".join(
        str(value)
        for value in (
            request.get("difficulty", ""),
            request.get("prompt", ""),
            request.get("trigger_safety_level", ""),
            request.get("quality_mode", ""),
            "Geometry Dash trigger schema motif drop buildup",
        )
        if value
    )
    chunks = retrieve_relevant_dataset_chunks(dataset_index, query, top_k=8)
    refs = dataset_index.reference_levels[:8]
    motifs: list[dict[str, Any]] = []
    for ref in refs:
        for motif in ref.get("common_motif_patterns", [])[:4]:
            if isinstance(motif, dict):
                motifs.append(motif)
    stats = dict(dataset_index.scan_result or {})
    stats.update(
        {
            "document_count": len(dataset_index.documents),
            "chunk_count": len(dataset_index.chunks),
            "reference_level_count": len(dataset_index.reference_levels),
            "example_count": len(dataset_index.examples),
        }
    )
    return DatasetContextForGeneration(
        relevant_docs_summary=summarize_dataset_context_for_prompt(chunks, max_chars=max_chars),
        relevant_motifs=motifs[:12],
        trigger_schema_summary=summarize_dataset_context_for_prompt(
            [{"title": item.get("file_name", item.get("relative_path", "trigger_schema")), "text": item.get("text_summary", "")} for item in dataset_index.trigger_schema_docs[:4]],
            max_chars=max_chars // 4,
        ),
        gd_logic_summary=summarize_dataset_context_for_prompt(
            [{"title": item.get("file_name", item.get("relative_path", "gd_logic")), "text": item.get("text_summary", "")} for item in dataset_index.gd_logic_docs[:4]],
            max_chars=max_chars // 4,
        ),
        style_memory_summary=_style_memory_from_dataset(dataset_index),
        failure_patterns_summary=[str(item.get("failure", "")) for item in dataset_index.examples if item.get("failure")][:8],
        success_patterns_summary=[str(item.get("success", "")) for item in dataset_index.examples if item.get("success")][:8],
        dataset_stats_summary=stats,
    )


def dataset_context_chunk(index: DatasetIndex, config: dict[str, Any], *, max_chars: int = 5000) -> dict[str, Any] | None:
    if not index.documents and not index.reference_levels and not index.chunks:
        return None
    context = build_generation_context_from_dataset(index, config, max_chars=max_chars)
    payload = context.to_dict()
    return {
        "source": "dataset_context",
        "path": "dataset_context",
        "title": "Dataset context summary",
        "kind": "dataset_context",
        "text": json.dumps(sanitize_learning_payload(payload), ensure_ascii=False, sort_keys=True),
    }


def _iter_supported_files(root: Path, *, recursive: bool, max_file_size_mb: float) -> list[Path]:
    scan = scan_dataset_dir(root, recursive=recursive, max_file_size_mb=max_file_size_mb)
    max_bytes = int(max(0.1, float(max_file_size_mb)) * 1024 * 1024)
    iterator = root.rglob("*") if recursive else root.glob("*")
    files: list[Path] = []
    for path in sorted(iterator):
        if not path.is_file() or _should_skip_path(path, root):
            continue
        if path.suffix.lower() not in SUPPORTED_DATASET_EXTENSIONS:
            continue
        if _safe_file_size(path) > max_bytes:
            continue
        files.append(path)
    if scan.files_indexed == 0:
        return []
    return files


def _read_supported_document(path: Path, *, max_chars: int) -> tuple[str, list[str]]:
    warnings: list[str] = []
    suffix = path.suffix.lower()
    if suffix == ".gmd":
        summary = summarize_reference_level(path)
        learned = learn_from_file(path)
        payload = {
            "reference_summary": summary,
            "learned_features": {
                "learned_level_count": len(learned.learned_levels),
                "motif_count": len(learned.motif_bank),
                "object_distributions": learned.object_distributions,
                "trigger_distributions": learned.trigger_distributions,
                "density_profiles": learned.density_profiles,
                "failure_patterns": learned.failure_patterns[:4],
            },
        }
        return json.dumps(sanitize_learning_payload(payload), ensure_ascii=False, sort_keys=True), warnings
    if suffix == ".csv":
        try:
            with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
                rows = list(csv.reader(handle))[:80]
            return "\n".join(",".join(row[:12]) for row in rows)[:max_chars], warnings
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"csv_read_failed:{path.name}:{exc}")
            return "", warnings
    text = path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".json":
        try:
            payload = json.loads(text)
            text = json.dumps(sanitize_learning_payload(_compact_json_payload(payload)), ensure_ascii=False, sort_keys=True)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"json_parse_failed:{path.name}:{exc}")
    elif suffix == ".jsonl":
        lines = []
        for line in text.splitlines()[:80]:
            try:
                lines.append(json.dumps(sanitize_learning_payload(_compact_json_payload(json.loads(line))), ensure_ascii=False, sort_keys=True))
            except Exception:
                if line.strip():
                    lines.append(line[:500])
        text = "\n".join(lines)
    return text[:max_chars], warnings


def _summarize_document_text(path: Path, text: str, *, document_type: str, max_chars: int) -> str:
    if document_type == "reference_level":
        return text[:max_chars]
    header = f"type={document_type} file={path.name}\n"
    return (header + text)[:max_chars]


def _reference_summary(path: Path, root: Path) -> dict[str, Any]:
    try:
        summary = summarize_reference_level(path)
    except Exception as exc:  # noqa: BLE001
        return {"name": path.name, "relative_path": _relative(path, root), "error": str(exc), "object_count": 0}
    summary["relative_path"] = _relative(path, root)
    return sanitize_learning_payload(summary)


def _example_summary(path: Path, document: DatasetDocument) -> dict[str, Any]:
    lowered = document.text_summary.lower()
    return {
        "source_name": path.name,
        "relative_path": document.relative_path,
        "document_type": document.document_type,
        "success": "score" if "score" in lowered and "failed" not in lowered else "",
        "failure": "failed" if "failed" in lowered or "error" in lowered else "",
        "summary": document.text_summary[:1200],
    }


def _document_metadata(document: DatasetDocument) -> dict[str, Any]:
    payload = document.to_dict()
    payload.pop("chunks", None)
    return sanitize_learning_payload(payload)


def _style_memory_from_dataset(index: DatasetIndex) -> dict[str, Any]:
    object_counts: dict[str, int] = {}
    trigger_counts: dict[str, int] = {}
    densities: list[float] = []
    for ref in index.reference_levels:
        for key, value in dict(ref.get("object_id_distribution", {})).items():
            object_counts[str(key)] = object_counts.get(str(key), 0) + int(value)
        for key, value in dict(ref.get("trigger_type_distribution", {})).items():
            trigger_counts[str(key)] = trigger_counts.get(str(key), 0) + int(value)
        if ref.get("average_density") is not None:
            densities.append(float(ref.get("average_density", 0.0) or 0.0))
    return {
        "common_object_ids": [key for key, _value in sorted(object_counts.items(), key=lambda item: item[1], reverse=True)[:24]],
        "common_trigger_types": [key for key, _value in sorted(trigger_counts.items(), key=lambda item: item[1], reverse=True)[:16]],
        "average_density": round(sum(densities) / max(1, len(densities)), 4) if densities else 0.0,
        "reference_level_count": len(index.reference_levels),
    }


def _dataset_hash(root: Path, documents: list[dict[str, Any]]) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(root).encode("utf-8", errors="replace"))
    for document in sorted(documents, key=lambda item: str(item.get("relative_path", ""))):
        hasher.update(str(document.get("relative_path", "")).encode("utf-8", errors="replace"))
        hasher.update(str(document.get("content_hash", "")).encode("utf-8", errors="replace"))
        hasher.update(str(document.get("size_bytes", "")).encode("utf-8", errors="replace"))
    return hasher.hexdigest()


def _compact_json_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        allowed = {}
        for key, value in payload.items():
            if str(key).lower() in {"level_string", "save_string", "raw_audio", "audio_bytes"}:
                allowed[key] = f"[omitted:{len(str(value))} chars]"
            elif len(allowed) < 80:
                allowed[key] = _compact_json_payload(value)
        return allowed
    if isinstance(payload, list):
        return [_compact_json_payload(item) for item in payload[:80]]
    if isinstance(payload, str) and len(payload) > 1000:
        return payload[:1000] + "...[truncated]"
    return payload


def _should_skip_path(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    parts = set(rel.parts[:-1])
    if any(part.startswith(".") for part in rel.parts):
        return True
    return bool(parts & SKIPPED_DIR_NAMES)


def _safe_file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except Exception:
        return 0


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except Exception:
        return path.name


def _inc(mapping: dict[str, int], key: str) -> None:
    mapping[key] = int(mapping.get(key, 0)) + 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
