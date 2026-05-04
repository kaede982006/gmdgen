# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from gmdgen.features.tokenizer import extract_object_id, extract_object_number
from gmdgen.representation.object_classifier import ObjectClass, classify


@dataclass(slots=True)
class Motif:
    motif_id: str
    source_level: str
    start_x: float
    end_x: float
    length_x: float
    section_type_hint: str
    roles: list[str] = field(default_factory=list)
    object_ids: list[str] = field(default_factory=list)
    trigger_types: list[str] = field(default_factory=list)
    group_usage: int = 0
    density: float = 0.0
    difficulty_hint: float = 0.5
    rhythm_hint: str = "steady"
    style_tags: list[str] = field(default_factory=list)
    compact_plan_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MotifRetrievalResult:
    query: str
    selected_motifs: list[Motif] = field(default_factory=list)
    rejected_motifs: list[Motif] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    empty_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MotifContext:
    section_id: int
    section_type: str
    selected_motifs: list[Motif] = field(default_factory=list)
    motif_summaries: list[dict[str, Any]] = field(default_factory=list)
    fallback_used: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def empty_motif_context(section_id: int, section_type: str, reason: str = "no_motifs_available") -> MotifContext:
    return MotifContext(
        section_id=section_id,
        section_type=section_type,
        fallback_used=True,
        warnings=[f"fallback_used: {reason}"],
        motif_summaries=[{"compact_plan_summary": "No reference motifs available; use safe generic structure constraints."}]
    )


@dataclass(slots=True)
class MotifBank:
    motifs: list[Motif] = field(default_factory=list)
    style_tags: list[str] = field(default_factory=list)
    density_profiles: dict[str, float] = field(default_factory=dict)
    trigger_patterns: dict[str, int] = field(default_factory=dict)
    gameplay_patterns: dict[str, int] = field(default_factory=dict)
    decoration_patterns: dict[str, int] = field(default_factory=dict)

    def retrieve(
        self,
        *,
        section_type: str,
        limit: int = 6,
        prefer_high_density: bool | None = None,
    ) -> list[Motif]:
        motifs = list(self.motifs)
        if prefer_high_density is None:
            prefer_high_density = section_type == "drop"
        def key(motif: Motif) -> tuple[float, float]:
            type_bonus = 1.0 if motif.section_type_hint == section_type else 0.0
            density_score = motif.density if prefer_high_density else 1.0 - motif.density
            return (type_bonus, density_score)
        return sorted(motifs, key=key, reverse=True)[: max(0, limit)]

    def to_prompt_summaries(self, *, section_type: str, limit: int = 6) -> list[dict[str, Any]]:
        return [
            {
                "motif_id": motif.motif_id,
                "section_type_hint": motif.section_type_hint,
                "length_x": round(motif.length_x, 2),
                "density": round(motif.density, 4),
                "roles": motif.roles[:8],
                "object_ids": motif.object_ids[:12],
                "trigger_types": motif.trigger_types[:8],
                "style_tags": motif.style_tags[:8],
                "compact_plan_summary": motif.compact_plan_summary,
            }
            for motif in self.retrieve(section_type=section_type, limit=limit)
        ]

    def inject_motif(self, target_x: float, section_type: str, *, difficulty: float = 0.5) -> list[dict[str, Any]]:
        motifs = self.retrieve(section_type=section_type, limit=3)
        if not motifs:
            return []
        import random
        selected = random.choice(motifs)
        # We would parse selected.compact_plan_summary or raw to actual ObjectPlans here
        # Returning stub mapping for now since raw objects aren't stored in Motif structure directly
        return [{"role": "motif_injected", "x": target_x, "original_id": selected.motif_id}]


def build_motif_bank_from_files(paths: Iterable[str | Path], *, window_x: float = 480.0) -> MotifBank:
    motifs: list[Motif] = []
    for path in paths:
        file_path = Path(path)
        if not file_path.exists() or not file_path.is_file():
            continue
        motifs.extend(extract_motifs_from_level(file_path.read_text(encoding="utf-8", errors="ignore"), source_level=file_path.name, window_x=window_x))
    return build_motif_bank(motifs)


def extract_motifs_from_level(level_string: str, *, source_level: str = "inline", window_x: float = 480.0) -> list[Motif]:
    objects = []
    for raw in level_string.split(";"):
        object_id = extract_object_id(raw)
        x = extract_object_number(raw, "2")
        if object_id is None or x is None:
            continue
        objects.append((float(x), str(object_id), raw))
    if not objects:
        return []
    objects.sort(key=lambda item: item[0])
    min_x = objects[0][0]
    max_x = objects[-1][0]
    motifs: list[Motif] = []
    index = 0
    start = min_x
    while start <= max_x:
        end = start + window_x
        chunk = [item for item in objects if start <= item[0] < end]
        if len(chunk) >= 2:
            motifs.append(_build_motif(chunk, source_level=source_level, motif_id=f"{source_level}:{index}", start_x=start, end_x=end))
            index += 1
        start += window_x * 0.65
    return motifs


def build_motif_bank(motifs: Iterable[Motif]) -> MotifBank:
    motif_list = list(motifs)
    trigger_counter: Counter[str] = Counter()
    gameplay_counter: Counter[str] = Counter()
    decoration_counter: Counter[str] = Counter()
    density_by_type: dict[str, list[float]] = {}
    tags: set[str] = set()
    for motif in motif_list:
        trigger_counter.update(motif.trigger_types)
        for role in motif.roles:
            if "decor" in role:
                decoration_counter[role] += 1
            elif role in {"orb", "pad", "obstacle", "portal"}:
                gameplay_counter[role] += 1
        density_by_type.setdefault(motif.section_type_hint, []).append(motif.density)
        tags.update(motif.style_tags)
    return MotifBank(
        motifs=motif_list,
        style_tags=sorted(tags),
        density_profiles={key: sum(values) / len(values) for key, values in density_by_type.items()},
        trigger_patterns=dict(trigger_counter),
        gameplay_patterns=dict(gameplay_counter),
        decoration_patterns=dict(decoration_counter),
    )


def retrieve_motifs_for_section(bank: MotifBank, section_type: str, *, limit: int = 6) -> list[dict[str, Any]]:
    return bank.to_prompt_summaries(section_type=section_type, limit=limit)

def build_motif_context_for_section(bank: MotifBank | None, section_id: int, section_type: str, limit: int = 6) -> MotifContext:
    if bank is None:
        return empty_motif_context(section_id, section_type, "no_motif_bank_provided")
    
    motifs = bank.retrieve(section_type=section_type, limit=limit)
    if not motifs:
        return empty_motif_context(section_id, section_type, "no_motifs_found_in_bank")
        
    summaries = [
        {
            "motif_id": m.motif_id,
            "section_type_hint": m.section_type_hint,
            "length_x": round(m.length_x, 2),
            "density": round(m.density, 4),
            "roles": m.roles[:8],
            "object_ids": m.object_ids[:12],
            "trigger_types": m.trigger_types[:8],
            "style_tags": m.style_tags[:8],
            "compact_plan_summary": m.compact_plan_summary,
        }
        for m in motifs
    ]
    return MotifContext(
        section_id=section_id,
        section_type=section_type,
        selected_motifs=motifs,
        motif_summaries=summaries,
        fallback_used=False
    )


def _build_motif(
    chunk: list[tuple[float, str, str]],
    *,
    source_level: str,
    motif_id: str,
    start_x: float,
    end_x: float,
) -> Motif:
    object_ids = [object_id for _, object_id, _ in chunk]
    roles = [_role_for_object_id(object_id) for object_id in object_ids]
    trigger_types = [object_id for object_id, role in zip(object_ids, roles) if role == "trigger"]
    density = min(1.0, len(chunk) / max(1.0, (end_x - start_x) / 30.0))
    section_type_hint = _section_type_hint(density, trigger_types, roles)
    style_tags = _style_tags(roles, density, trigger_types)
    return Motif(
        motif_id=motif_id,
        source_level=source_level,
        start_x=start_x,
        end_x=end_x,
        length_x=end_x - start_x,
        section_type_hint=section_type_hint,
        roles=sorted(set(roles)),
        object_ids=object_ids[:32],
        trigger_types=trigger_types[:16],
        group_usage=sum(1 for _, _, raw in chunk if ",57," in raw or ",51," in raw),
        density=round(density, 4),
        difficulty_hint=0.75 if "obstacle" in roles else 0.45,
        rhythm_hint="dense" if density > 0.65 else "sparse" if density < 0.25 else "steady",
        style_tags=style_tags,
        compact_plan_summary=f"{section_type_hint} motif with {len(chunk)} objects, roles={sorted(set(roles))[:5]}",
    )


def _role_for_object_id(object_id: str) -> str:
    try:
        object_class = classify(object_id)
    except Exception:
        return "unknown"
    if object_class == ObjectClass.TRIGGER:
        return "trigger"
    if object_class == ObjectClass.PORTAL:
        return "portal"
    if object_class == ObjectClass.DECORATION:
        return "decoration"
    if object_class == ObjectClass.STRUCTURE:
        return "structure"
    if object_id in {"8", "39"}:
        return "obstacle"
    if object_id in {"35", "36", "84", "140", "141", "1022"}:
        return "orb" if object_id != "35" and object_id != "140" else "pad"
    return object_class.value


def _section_type_hint(density: float, trigger_types: list[str], roles: list[str]) -> str:
    if density >= 0.65 and trigger_types:
        return "drop"
    if density >= 0.45:
        return "buildup"
    if density <= 0.2:
        return "break"
    if "portal" in roles:
        return "transition"
    return "normal"


def _style_tags(roles: list[str], density: float, trigger_types: list[str]) -> list[str]:
    tags = []
    if density >= 0.65:
        tags.append("high_density")
    if trigger_types:
        tags.append("trigger_accent")
    if roles.count("decoration") >= roles.count("structure"):
        tags.append("decoration_heavy")
    if "obstacle" in roles:
        tags.append("hazard_gameplay")
    return tags or ["minimal"]
