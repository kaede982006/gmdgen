from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from gmdgen.data.preprocess import split_level_objects
from gmdgen.features.tokenizer import extract_object_field, extract_object_id, extract_object_number
from gmdgen.io.gmd_decoder import decode_level_data
from gmdgen.io.gmd_parser import parse_gmd_file
from gmdgen.representation.object_classifier import classify


@dataclass(slots=True)
class ContextDocument:
    path: str
    title: str
    text: str
    kind: str


@dataclass(slots=True)
class ContextChunk:
    source: str
    title: str
    text: str
    score: float = 0.0

    def to_dict(self) -> dict:
        return {"source": self.source, "title": self.title, "text": self.text, "score": self.score}


def load_context_documents(context_dir: str | Path | None) -> list[ContextDocument]:
    if not context_dir:
        return []
    root = Path(context_dir)
    if not root.exists() or not root.is_dir():
        return []
    result: list[ContextDocument] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".json", ".py", ".gmd"}:
            continue
        try:
            if path.suffix.lower() == ".gmd":
                text = json.dumps(summarize_reference_level(path), ensure_ascii=False)
            else:
                text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        result.append(ContextDocument(path=str(path), title=path.name, text=text, kind=path.suffix.lower().lstrip(".")))
    return result


def summarize_context_documents(documents: Iterable[ContextDocument], max_chars: int) -> list[dict]:
    remaining = max(0, int(max_chars))
    chunks: list[dict] = []
    for doc in documents:
        if remaining <= 0:
            break
        text = doc.text[: min(len(doc.text), remaining)]
        chunks.append({"source": doc.path, "title": doc.title, "kind": doc.kind, "text": text})
        remaining -= len(text)
    return chunks


def load_reference_levels(reference_levels_dir: str | Path | None) -> list[dict]:
    if not reference_levels_dir:
        return []
    root = Path(reference_levels_dir)
    if not root.exists() or not root.is_dir():
        return []
    summaries = []
    for path in sorted(root.glob("*.gmd"))[:24]:
        summaries.append(summarize_reference_level(path))
    return summaries


def summarize_reference_level(level_string_or_file: str | Path) -> dict:
    path = Path(str(level_string_or_file))
    level_name = path.name if path.exists() else "inline_level"
    decoded = ""
    if path.exists() and path.is_file():
        try:
            parsed = parse_gmd_file(path)
            entry = parsed.tags.get("k4")
            decoded = decode_level_data(entry[1]) if entry else ""
        except Exception:  # noqa: BLE001
            decoded = path.read_text(encoding="utf-8", errors="replace")[:50000]
    else:
        decoded = str(level_string_or_file)
    objects = split_level_objects(decoded) if decoded else []
    object_ids: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    trigger_type_counts: dict[str, int] = {}
    color_channels: dict[str, int] = {}
    group_usage = 0
    x_values: list[float] = []
    high_detail = 0
    for obj in objects[:5000]:
        object_id = extract_object_id(obj)
        if not object_id:
            continue
        object_ids[object_id] = object_ids.get(object_id, 0) + 1
        cls = classify(object_id).value
        class_counts[cls] = class_counts.get(cls, 0) + 1
        role_counts[_role_hint(object_id, cls)] = role_counts.get(_role_hint(object_id, cls), 0) + 1
        if cls == "trigger":
            trigger_type_counts[object_id] = trigger_type_counts.get(object_id, 0) + 1
        if ",155," in obj:
            group_usage += 1
        if extract_object_field(obj, "21"):
            high_detail += 1
        color = extract_object_field(obj, "23") or extract_object_field(obj, "50")
        if color:
            color_channels[color] = color_channels.get(color, 0) + 1
        x = extract_object_number(obj, "2")
        if x is not None:
            x_values.append(float(x))
    trigger_count = class_counts.get("trigger", 0)
    portal_count = class_counts.get("portal", 0)
    object_count = len(objects)
    structure_count = class_counts.get("structure", 0)
    decoration_count = class_counts.get("decoration", 0)
    gameplay_count = role_counts.get("gameplay_orb", 0) + role_counts.get("gameplay_pad", 0) + role_counts.get("obstacle", 0)
    motifs = _extract_motif_summaries(objects[:1200])
    return {
        "name": level_name,
        "object_count": object_count,
        "trigger_count": trigger_count,
        "structure_object_ratio": structure_count / max(1, object_count),
        "decoration_object_ratio": decoration_count / max(1, object_count),
        "gameplay_object_ratio": gameplay_count / max(1, object_count),
        "trigger_ratio": trigger_count / max(1, object_count),
        "speed_portal_count": portal_count,
        "mode_portal_count": portal_count,
        "group_usage_count": group_usage,
        "color_channel_usage": dict(sorted(color_channels.items(), key=lambda item: item[1], reverse=True)[:12]),
        "high_detail_ratio": high_detail / max(1, object_count),
        "object_id_distribution": dict(sorted(object_ids.items(), key=lambda item: item[1], reverse=True)[:25]),
        "role_distribution": dict(sorted(role_counts.items(), key=lambda item: item[1], reverse=True)[:16]),
        "trigger_type_distribution": dict(sorted(trigger_type_counts.items(), key=lambda item: item[1], reverse=True)[:16]),
        "class_distribution": class_counts,
        "average_density": object_count / max(1.0, (max(x_values) - min(x_values)) / 1000.0) if x_values else 0.0,
        "density_by_section": _density_bins(x_values, object_count),
        "common_motif_patterns": motifs,
        "average_motif_length": sum(item["length_x"] for item in motifs) / len(motifs) if motifs else 0.0,
        "section_transition_style": "triggered" if trigger_count else "structure_only",
        "drop_density_multiplier": 1.0,
        "decoration_density_profile": "reference_summary",
        "text_included": False,
    }


class ContextRetriever:
    def retrieve(self, query: str, top_k: int) -> list[ContextChunk]:
        raise NotImplementedError


class LocalKeywordRetriever(ContextRetriever):
    def __init__(self, documents: list[ContextDocument]) -> None:
        self.documents = list(documents)

    def retrieve(self, query: str, top_k: int = 4) -> list[ContextChunk]:
        terms = {term.lower() for term in query.replace("_", " ").split() if term.strip()}
        scored: list[ContextChunk] = []
        for doc in self.documents:
            lowered = doc.text.lower()
            score = sum(lowered.count(term) for term in terms)
            if score <= 0:
                continue
            scored.append(ContextChunk(source=doc.path, title=doc.title, text=doc.text[:2000], score=float(score)))
        return sorted(scored, key=lambda chunk: chunk.score, reverse=True)[:top_k]


def _role_hint(object_id: str, cls: str) -> str:
    if object_id in {"35", "140"}:
        return "gameplay_pad"
    if object_id in {"36", "84", "141", "1022"}:
        return "gameplay_orb"
    if object_id in {"8", "39"}:
        return "obstacle"
    if cls == "trigger":
        return "trigger"
    if cls == "decoration":
        return "decoration"
    if cls == "structure":
        return "structure"
    return cls


def _density_bins(x_values: list[float], object_count: int, *, bins: int = 4) -> dict[str, float]:
    if not x_values or object_count <= 0:
        return {}
    start = min(x_values)
    span = max(1.0, max(x_values) - start)
    counts = {str(idx): 0 for idx in range(bins)}
    for x in x_values:
        idx = min(bins - 1, int(((x - start) / span) * bins))
        counts[str(idx)] += 1
    return {key: round(value / max(1, object_count), 4) for key, value in counts.items()}


def _extract_motif_summaries(objects: list[str], *, window: int = 8) -> list[dict]:
    motifs: list[dict] = []
    for start in range(0, len(objects), max(1, window)):
        chunk = objects[start:start + window]
        if not chunk:
            continue
        ids = [extract_object_id(obj) for obj in chunk if extract_object_id(obj)]
        xs = [extract_object_number(obj, "2") for obj in chunk]
        xs = [float(x) for x in xs if x is not None]
        if not ids or not xs:
            continue
        roles = [_role_hint(object_id, classify(object_id).value) for object_id in ids]
        length_x = max(xs) - min(xs) if len(xs) > 1 else 0.0
        motifs.append(
            {
                "motif_id": f"motif_{len(motifs)}",
                "object_count": len(chunk),
                "trigger_count": sum(1 for role in roles if role == "trigger"),
                "length_x": round(length_x, 2),
                "density": round(len(chunk) / max(1.0, length_x / 100.0), 4),
                "roles": sorted(set(roles)),
                "object_ids": ids[:12],
                "section_type_hint": "drop" if len(chunk) >= window and length_x < 360 else "normal",
                "style_tags": sorted(set(roles))[:6],
            }
        )
        if len(motifs) >= 8:
            break
    return motifs
