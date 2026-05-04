from __future__ import annotations

import json
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from gmdgen.features.tokenizer import extract_object_field, extract_object_id, extract_object_number
from gmdgen.generate.style_bank import Motif, build_motif_bank, extract_motifs_from_level
from gmdgen.learning.store import DEFAULT_LEARNING_DIR, sanitize_learning_payload
from gmdgen.representation.object_classifier import ObjectClass, classify


DEFAULT_LEARNED_DATA_DIR = DEFAULT_LEARNING_DIR.parent / "learned_data"


@dataclass(slots=True)
class LevelFeatureSummary:
    source_name: str
    source_type: str
    object_count: int = 0
    trigger_count: int = 0
    object_id_distribution: dict[str, int] = field(default_factory=dict)
    trigger_type_distribution: dict[str, int] = field(default_factory=dict)
    role_distribution: dict[str, int] = field(default_factory=dict)
    group_usage_count: int = 0
    color_channel_usage: dict[str, int] = field(default_factory=dict)
    speed_portal_count: int = 0
    density_by_x: dict[str, int] = field(default_factory=dict)
    density_by_section: dict[str, float] = field(default_factory=dict)
    structure_ratio: float = 0.0
    decoration_ratio: float = 0.0
    gameplay_ratio: float = 0.0
    trigger_ratio: float = 0.0
    high_detail_ratio: float = 0.0
    estimated_difficulty: float = 0.5
    style_tags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MotifFeature:
    motif_id: str
    source_name: str
    start_x: float
    end_x: float
    length_x: float
    object_count: int
    trigger_count: int
    density: float
    object_ids: list[str] = field(default_factory=list)
    trigger_types: list[str] = field(default_factory=list)
    role_pattern: list[str] = field(default_factory=list)
    section_type_hint: str = "normal"
    energy_hint: str = "medium"
    difficulty_hint: float = 0.5
    style_tags: list[str] = field(default_factory=list)
    compact_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StyleProfile:
    profile_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    source_count: int = 0
    common_object_ids: list[str] = field(default_factory=list)
    common_trigger_types: list[str] = field(default_factory=list)
    preferred_density_range: tuple[float, float] = (0.0, 0.0)
    decoration_ratio: float = 0.0
    gameplay_ratio: float = 0.0
    trigger_ratio: float = 0.0
    drop_density_multiplier: float = 1.0
    buildup_density_curve: list[float] = field(default_factory=list)
    common_motif_patterns: list[str] = field(default_factory=list)
    color_usage_summary: dict[str, int] = field(default_factory=dict)
    speed_usage_summary: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["preferred_density_range"] = list(self.preferred_density_range)
        return payload


@dataclass(slots=True)
class LearnedDataStore:
    learned_levels: list[dict[str, Any]] = field(default_factory=list)
    style_profiles: list[dict[str, Any]] = field(default_factory=list)
    motif_bank: list[dict[str, Any]] = field(default_factory=list)
    object_distributions: dict[str, int] = field(default_factory=dict)
    trigger_distributions: dict[str, int] = field(default_factory=dict)
    density_profiles: dict[str, float] = field(default_factory=dict)
    section_profiles: dict[str, Any] = field(default_factory=dict)
    feedback_examples: list[dict[str, Any]] = field(default_factory=list)
    failure_patterns: list[str] = field(default_factory=list)
    success_patterns: list[str] = field(default_factory=list)
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def learned_data_store_path(store_dir: str | Path | None = None) -> Path:
    directory = Path(store_dir) if store_dir else DEFAULT_LEARNED_DATA_DIR
    return directory.expanduser() / "learned_data.json"


def learn_from_file(path: str | Path) -> LearnedDataStore:
    file_path = Path(path).expanduser()
    store = LearnedDataStore(updated_at=_now())
    if not file_path.exists() or not file_path.is_file():
        store.failure_patterns.append(f"missing_or_not_file:{file_path.name}")
        return store
    try:
        level_string = _read_level_like_file(file_path)
        if not level_string.strip():
            store.failure_patterns.append(f"empty_or_unsupported:{file_path.name}")
            return store
        summary = extract_level_features(level_string, source_name=file_path.name, source_type=file_path.suffix.lower().lstrip(".") or "level")
        motifs = extract_motifs(level_string, source_name=file_path.name)
        if summary.object_count <= 0 and summary.trigger_count <= 0:
            store.failure_patterns.append(f"no_level_objects:{file_path.name}")
            return store
        profile = build_style_profile([summary])
        store.learned_levels.append(summary.to_dict())
        store.motif_bank.extend(motif.to_dict() for motif in motifs)
        store.style_profiles.append(profile.to_dict())
        store.object_distributions = dict(summary.object_id_distribution)
        store.trigger_distributions = dict(summary.trigger_type_distribution)
        store.density_profiles = {"average": _average_density(summary)}
        store.success_patterns.append(f"learned:{file_path.name}")
        return store
    except Exception as exc:  # noqa: BLE001
        store.failure_patterns.append(f"learning_failed:{file_path.name}:{exc}")
        return store


def learn_from_directory(path: str | Path) -> LearnedDataStore:
    directory = Path(path).expanduser()
    store = LearnedDataStore(updated_at=_now())
    if not directory.exists() or not directory.is_dir():
        store.failure_patterns.append(f"missing_or_not_directory:{directory.name}")
        return store
    for file_path in sorted(directory.rglob("*")):
        if file_path.suffix.lower() not in {".gmd", ".txt", ".json"} or not file_path.is_file():
            continue
        child = learn_from_file(file_path)
        store = update_learned_data_store(store, child)
    return store


def extract_level_features(
    level_string_or_file: str | Path,
    *,
    source_name: str = "inline",
    source_type: str = "gmd",
) -> LevelFeatureSummary:
    maybe_path = None
    try:
        maybe_path = Path(str(level_string_or_file))
    except Exception:
        maybe_path = None
    if isinstance(level_string_or_file, Path) or (maybe_path is not None and len(str(level_string_or_file)) < 260 and maybe_path.exists()):
        path = Path(level_string_or_file)
        source_name = path.name
        source_type = path.suffix.lower().lstrip(".") or source_type
        level_string = _read_level_like_file(path)
    else:
        level_string = str(level_string_or_file)
    objects = _objects_from_level_string(level_string)
    object_ids: Counter[str] = Counter()
    triggers: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    density: Counter[str] = Counter()
    colors: Counter[str] = Counter()
    groups: set[int] = set()
    x_values: list[float] = []
    trigger_count = 0
    speed_portals = 0
    high_detail = 0
    for raw in objects:
        object_id = extract_object_id(raw)
        x = extract_object_number(raw, "2")
        if not object_id:
            continue
        object_ids[str(object_id)] += 1
        role = _role_for_object_id(str(object_id))
        roles[role] += 1
        if role == "trigger":
            triggers[str(object_id)] += 1
            trigger_count += 1
        if role == "speed_portal":
            speed_portals += 1
        if x is not None:
            x_values.append(float(x))
            density[str(int(float(x) // 480))] += 1
        group_raw = extract_object_field(raw, "57") or extract_object_field(raw, "155")
        if group_raw:
            for item in str(group_raw).replace(".", ",").split(","):
                if item.strip().isdigit():
                    groups.add(int(item.strip()))
        color_raw = extract_object_field(raw, "21") or extract_object_field(raw, "50")
        if color_raw:
            colors[str(color_raw)] += 1
        if len(raw.split(",")) > 12:
            high_detail += 1
    total = max(1, len(objects))
    density_by_section = {bucket: round(count / max(1, total), 4) for bucket, count in density.items()}
    style_tags = _style_tags_from_counts(roles, trigger_count, total)
    return LevelFeatureSummary(
        source_name=source_name,
        source_type=source_type,
        object_count=max(0, len(objects) - trigger_count),
        trigger_count=trigger_count,
        object_id_distribution=dict(object_ids),
        trigger_type_distribution=dict(triggers),
        role_distribution=dict(roles),
        group_usage_count=len(groups),
        color_channel_usage=dict(colors),
        speed_portal_count=speed_portals,
        density_by_x=dict(density),
        density_by_section=density_by_section,
        structure_ratio=roles.get("structure", 0) / total,
        decoration_ratio=roles.get("decoration", 0) / total,
        gameplay_ratio=(roles.get("orb", 0) + roles.get("pad", 0) + roles.get("obstacle", 0) + roles.get("portal", 0)) / total,
        trigger_ratio=trigger_count / total,
        high_detail_ratio=high_detail / total,
        estimated_difficulty=min(1.0, (roles.get("obstacle", 0) + roles.get("orb", 0) + roles.get("pad", 0)) / total + speed_portals * 0.03),
        style_tags=style_tags,
    )


def extract_motifs(level_summary_or_string: LevelFeatureSummary | str, *, source_name: str = "inline") -> list[MotifFeature]:
    if isinstance(level_summary_or_string, LevelFeatureSummary):
        return []
    motifs = extract_motifs_from_level(str(level_summary_or_string), source_level=source_name)
    return [_motif_to_feature(motif) for motif in motifs]


def build_style_profile(level_summaries: Iterable[LevelFeatureSummary]) -> StyleProfile:
    summaries = list(level_summaries)
    object_ids: Counter[str] = Counter()
    triggers: Counter[str] = Counter()
    colors: Counter[str] = Counter()
    densities: list[float] = []
    deco = gameplay = trig = 0.0
    for summary in summaries:
        object_ids.update(summary.object_id_distribution)
        triggers.update(summary.trigger_type_distribution)
        colors.update(summary.color_channel_usage)
        densities.append(_average_density(summary))
        deco += summary.decoration_ratio
        gameplay += summary.gameplay_ratio
        trig += summary.trigger_ratio
    count = max(1, len(summaries))
    density_min = min(densities, default=0.0)
    density_max = max(densities, default=0.0)
    return StyleProfile(
        source_count=len(summaries),
        common_object_ids=[key for key, _count in object_ids.most_common(24)],
        common_trigger_types=[key for key, _count in triggers.most_common(12)],
        preferred_density_range=(round(density_min, 4), round(density_max, 4)),
        decoration_ratio=round(deco / count, 4),
        gameplay_ratio=round(gameplay / count, 4),
        trigger_ratio=round(trig / count, 4),
        drop_density_multiplier=1.35 if density_max > density_min else 1.0,
        buildup_density_curve=[round(value, 4) for value in sorted(densities)[:8]],
        common_motif_patterns=[f"objects:{key}" for key, _count in object_ids.most_common(8)],
        color_usage_summary=dict(colors.most_common(12)),
        speed_usage_summary={"speed_portal_count": sum(item.speed_portal_count for item in summaries)},
    )


def update_learned_data_store(base: LearnedDataStore | None, features: LearnedDataStore | LevelFeatureSummary) -> LearnedDataStore:
    store = base or LearnedDataStore()
    if isinstance(features, LevelFeatureSummary):
        incoming = LearnedDataStore(
            learned_levels=[features.to_dict()],
            style_profiles=[build_style_profile([features]).to_dict()],
            object_distributions=dict(features.object_id_distribution),
            trigger_distributions=dict(features.trigger_type_distribution),
            density_profiles={"average": _average_density(features)},
        )
    else:
        incoming = features
    store.learned_levels.extend(incoming.learned_levels)
    store.style_profiles.extend(incoming.style_profiles)
    store.motif_bank.extend(incoming.motif_bank)
    _merge_counter_dict(store.object_distributions, incoming.object_distributions)
    _merge_counter_dict(store.trigger_distributions, incoming.trigger_distributions)
    store.failure_patterns.extend(incoming.failure_patterns)
    store.success_patterns.extend(incoming.success_patterns)
    densities = [float(store.density_profiles.get("average", 0.0)), float(incoming.density_profiles.get("average", 0.0))]
    store.density_profiles["average"] = round(sum(densities) / max(1, len([d for d in densities if d > 0])), 4)
    if store.learned_levels:
        summaries = [_summary_from_dict(item) for item in store.learned_levels]
        store.style_profiles = [build_style_profile([item for item in summaries if item is not None]).to_dict()]
    store.updated_at = _now()
    return store


def save_learned_data_store(store: LearnedDataStore | dict[str, Any], *, store_dir: str | Path | None = None) -> Path:
    path = learned_data_store_path(store_dir)
    payload = store.to_dict() if isinstance(store, LearnedDataStore) else dict(store)
    sanitized = sanitize_learning_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitized, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_learned_data_store(*, store_dir: str | Path | None = None) -> LearnedDataStore:
    path = learned_data_store_path(store_dir)
    if not path.exists():
        return LearnedDataStore()
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return LearnedDataStore(failure_patterns=["corrupt_learned_data_store"])
    return LearnedDataStore(
        learned_levels=list(payload.get("learned_levels", [])),
        style_profiles=list(payload.get("style_profiles", [])),
        motif_bank=list(payload.get("motif_bank", [])),
        object_distributions=dict(payload.get("object_distributions", {})),
        trigger_distributions=dict(payload.get("trigger_distributions", {})),
        density_profiles=dict(payload.get("density_profiles", {})),
        section_profiles=dict(payload.get("section_profiles", {})),
        feedback_examples=list(payload.get("feedback_examples", [])),
        failure_patterns=list(payload.get("failure_patterns", [])),
        success_patterns=list(payload.get("success_patterns", [])),
        updated_at=str(payload.get("updated_at", "")),
    )


def clear_learned_data_store(*, store_dir: str | Path | None = None) -> bool:
    path = learned_data_store_path(store_dir)
    if path.exists():
        path.unlink()
        return True
    return False


def summarize_learned_data_for_prompt(
    store: LearnedDataStore | None = None,
    *,
    store_dir: str | Path | None = None,
    max_chars: int = 4000,
) -> dict[str, Any]:
    store = store or load_learned_data_store(store_dir=store_dir)
    style_profile = store.style_profiles[0] if store.style_profiles else {}
    payload = {
        "learned_level_count": len(store.learned_levels),
        "motif_count": len(store.motif_bank),
        "learned_style_summary": style_profile,
        "learned_object_distribution": dict(list(store.object_distributions.items())[:24]),
        "learned_trigger_distribution": dict(list(store.trigger_distributions.items())[:16]),
        "learned_density_profile": dict(store.density_profiles),
        "learned_failure_patterns": store.failure_patterns[:12],
        "learned_success_patterns": store.success_patterns[:12],
        "retrieved_motifs": store.motif_bank[:8],
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if len(text) <= max_chars:
        return payload
    payload["retrieved_motifs"] = payload["retrieved_motifs"][:3]
    payload["learned_failure_patterns"] = payload["learned_failure_patterns"][:5]
    payload["learned_success_patterns"] = payload["learned_success_patterns"][:5]
    return payload


def retrieve_motifs_for_section(section_task: dict[str, Any], learned_store: LearnedDataStore, *, limit: int = 6) -> list[dict[str, Any]]:
    section_type = str(section_task.get("section_type", "normal"))
    motifs = []
    for item in learned_store.motif_bank:
        type_bonus = 1 if item.get("section_type_hint") == section_type else 0
        density = float(item.get("density", 0.0) or 0.0)
        motifs.append((type_bonus, density if section_type == "drop" else 1.0 - density, item))
    return [item for _type_bonus, _density, item in sorted(motifs, reverse=True)[:limit]]


def export_finetune_jsonl_from_learning_store(output_path: str | Path, *, store_dir: str | Path | None = None, min_rating: int = 4) -> Path:
    store = load_learned_data_store(store_dir=store_dir)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for level in store.learned_levels:
        if int(level.get("user_rating", min_rating) or min_rating) < min_rating:
            continue
        lines.append(json.dumps({"input": summarize_learned_data_for_prompt(store), "output": level}, ensure_ascii=False, sort_keys=True))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def export_preference_pairs_from_learning_store(output_path: str | Path, *, store_dir: str | Path | None = None) -> Path:
    store = load_learned_data_store(store_dir=store_dir)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    chosen = [item for item in store.learned_levels if int(item.get("user_rating", 0) or 0) >= 4]
    rejected = [item for item in store.learned_levels if int(item.get("user_rating", 0) or 0) in {1, 2}]
    lines = []
    for good, bad in zip(chosen, rejected):
        lines.append(json.dumps({"chosen": good, "rejected": bad, "reason": "user_rating"}, ensure_ascii=False, sort_keys=True))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def _read_level_like_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except Exception:
            return ""
        for key in ("level_string", "save_string", "encoded_level", "final_encoded_plan"):
            value = payload.get(key) if isinstance(payload, dict) else None
            if isinstance(value, str):
                return value
        if isinstance(payload, dict):
            return _objects_to_level_like_string(payload.get("objects") or payload.get("level_objects") or [])
        return ""
    return text


def _objects_from_level_string(level_string: str) -> list[str]:
    return [part.strip() for part in str(level_string).split(";") if extract_object_id(part.strip())]


def _objects_to_level_like_string(objects: Any) -> str:
    if not isinstance(objects, list):
        return ""
    return ";".join(str(item) for item in objects if isinstance(item, str))


def _motif_to_feature(motif: Motif) -> MotifFeature:
    return MotifFeature(
        motif_id=motif.motif_id,
        source_name=motif.source_level,
        start_x=motif.start_x,
        end_x=motif.end_x,
        length_x=motif.length_x,
        object_count=len(motif.object_ids) - len(motif.trigger_types),
        trigger_count=len(motif.trigger_types),
        density=motif.density,
        object_ids=list(motif.object_ids),
        trigger_types=list(motif.trigger_types),
        role_pattern=list(motif.roles),
        section_type_hint=motif.section_type_hint,
        energy_hint="high" if motif.density > 0.65 else "low" if motif.density < 0.25 else "medium",
        difficulty_hint=motif.difficulty_hint,
        style_tags=list(motif.style_tags),
        compact_summary=motif.compact_plan_summary,
    )


def _role_for_object_id(object_id: str) -> str:
    if object_id in {"200", "201", "202", "203", "1334", "1335", "1346"}:
        return "speed_portal"
    if object_id in {"35", "140"}:
        return "pad"
    if object_id in {"36", "84", "141", "1022"}:
        return "orb"
    if object_id in {"8", "39"}:
        return "obstacle"
    try:
        cls = classify(object_id)
    except Exception:
        return "unknown"
    if cls == ObjectClass.TRIGGER:
        return "trigger"
    if cls == ObjectClass.DECORATION:
        return "decoration"
    if cls == ObjectClass.STRUCTURE:
        return "structure"
    if cls == ObjectClass.PORTAL:
        return "portal"
    return cls.value


def _style_tags_from_counts(roles: Counter[str], trigger_count: int, total: int) -> list[str]:
    tags: list[str] = []
    if roles.get("decoration", 0) / max(1, total) > 0.3:
        tags.append("decoration_heavy")
    if roles.get("structure", 0) / max(1, total) > 0.45:
        tags.append("structure_heavy")
    if trigger_count / max(1, total) > 0.15:
        tags.append("trigger_heavy")
    if roles.get("orb", 0) + roles.get("pad", 0) > 0:
        tags.append("gameplay_orb_pad")
    return tags or ["minimal"]


def _average_density(summary: LevelFeatureSummary) -> float:
    values = list(summary.density_by_section.values())
    if not values:
        return 0.0
    return round(sum(float(value) for value in values) / len(values), 4)


def _merge_counter_dict(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[str(key)] = int(target.get(str(key), 0)) + int(value)


def _summary_from_dict(payload: dict[str, Any]) -> LevelFeatureSummary | None:
    try:
        return LevelFeatureSummary(
            source_name=str(payload.get("source_name", "")),
            source_type=str(payload.get("source_type", "")),
            object_count=int(payload.get("object_count", 0)),
            trigger_count=int(payload.get("trigger_count", 0)),
            object_id_distribution=dict(payload.get("object_id_distribution", {})),
            trigger_type_distribution=dict(payload.get("trigger_type_distribution", {})),
            role_distribution=dict(payload.get("role_distribution", {})),
            group_usage_count=int(payload.get("group_usage_count", 0)),
            color_channel_usage=dict(payload.get("color_channel_usage", {})),
            speed_portal_count=int(payload.get("speed_portal_count", 0)),
            density_by_x=dict(payload.get("density_by_x", {})),
            density_by_section=dict(payload.get("density_by_section", {})),
            structure_ratio=float(payload.get("structure_ratio", 0.0)),
            decoration_ratio=float(payload.get("decoration_ratio", 0.0)),
            gameplay_ratio=float(payload.get("gameplay_ratio", 0.0)),
            trigger_ratio=float(payload.get("trigger_ratio", 0.0)),
            high_detail_ratio=float(payload.get("high_detail_ratio", 0.0)),
            estimated_difficulty=float(payload.get("estimated_difficulty", 0.5)),
            style_tags=list(payload.get("style_tags", [])),
            warnings=list(payload.get("warnings", [])),
        )
    except Exception:
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_palette_from_learned_store(
    store: "LearnedDataStore | None",
    *,
    min_occurrences: int = 1,
    max_per_class: int = 32,
) -> dict[str, list[str]]:
    """Build an `ids_by_class` palette from a learned data store.

    Returns a dict mapping ObjectClass.value -> list of object IDs sorted by
    learned frequency. Used by the materializer to prefer dataset-derived
    object IDs over the built-in safe palette when learning data exists.

    If the store is empty/None, returns an empty dict so callers fall back
    to their own safe defaults.
    """
    if store is None:
        return {}
    distribution = dict(getattr(store, "object_distributions", {}) or {})
    if not distribution:
        return {}

    by_class: dict[str, Counter[str]] = {}
    for object_id, count in distribution.items():
        if not isinstance(object_id, str) or not object_id:
            continue
        try:
            count_int = int(count)
        except (TypeError, ValueError):
            continue
        if count_int < min_occurrences:
            continue
        cls = classify(object_id).value
        by_class.setdefault(cls, Counter())[object_id] += count_int

    palette: dict[str, list[str]] = {}
    for cls, counter in by_class.items():
        ranked = [obj_id for obj_id, _ in counter.most_common(max_per_class)]
        if ranked:
            palette[cls] = ranked
    return palette


def palette_metrics(palette: dict[str, list[str]]) -> dict[str, Any]:
    """Quick metrics about a built palette for reporting and tests."""
    total_unique = len({oid for ids in palette.values() for oid in ids})
    return {
        "class_count": len(palette),
        "unique_object_id_count": total_unique,
        "ids_per_class": {cls: len(ids) for cls, ids in palette.items()},
    }
