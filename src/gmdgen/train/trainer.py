# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from gmdgen.data.loader import load_dataset_with_report
from gmdgen.data.preprocess import (
    extract_level_header,
    filter_records_by_object_count,
    objects_cross_portal,
    split_level_objects,
)
from gmdgen.data.schema import TagMap
from gmdgen.features.stats import object_token_frequencies
from gmdgen.features.tokenizer import (
    extract_object_id,
    extract_object_number,
    records_to_token_sequences,
)
from gmdgen.generate.scoring import build_reference_stats_from_records
from gmdgen.io.gmd_decoder import decode_level_description
from gmdgen.ml.architectures import build_audio_conditioned_model_spec
from gmdgen.representation.object_classifier import ObjectClass, classify
from gmdgen.models.markov_model import MarkovModel
from gmdgen.train.evaluator import evaluate_training_run
from gmdgen.utils.seed import set_global_seed


def _serialize_tags(tags: TagMap) -> list[dict[str, str]]:
    return [
        {"key": key, "type": value_type, "value": value}
        for key, (value_type, value) in tags.items()
    ]


def _optional_int(config: dict[str, Any], key: str, default: int | None) -> int | None:
    value = config.get(key, default)
    if value is None:
        return None
    return int(value)


def _int_or_none(value: float | None) -> int | None:
    if value is None:
        return None
    return int(round(value))


def _decode_record_description(record) -> str:
    k3_entry = record.document.tags.get("k3")
    if not k3_entry:
        return ""
    _, encoded_description = k3_entry
    return decode_level_description(encoded_description).strip()


def _build_generation_assets(
    sequences: list[list[str]],
    *,
    records: list,
    max_prototypes_per_id: int,
    max_y_samples_per_id: int,
    max_delta_samples: int,
    chunk_size: int,
    chunk_stride: int,
    max_chunks_per_level: int,
    min_chunk_objects: int,
) -> dict[str, Any]:
    object_prototypes: dict[str, list[str]] = defaultdict(list)
    object_y_samples: dict[str, list[int]] = defaultdict(list)
    delta_x_samples: list[int] = []
    level_profiles: list[dict[str, Any]] = []
    chunk_library: list[dict[str, Any]] = []

    for record in records:
        level_name = record.document.tags.get("k2", ("s", record.document.path.stem))[1]
        level_description = _decode_record_description(record)
        level_objects = split_level_objects(record.decoded_level_data)
        level_counter: Counter[str] = Counter()
        previous_x: int | None = None

        for level_object in level_objects:
            object_id = extract_object_id(level_object)
            if not object_id:
                continue

            level_counter[object_id] += 1

            if len(object_prototypes[object_id]) < max_prototypes_per_id:
                object_prototypes[object_id].append(level_object)

            y_value = _int_or_none(extract_object_number(level_object, "3"))
            if (
                y_value is not None
                and len(object_y_samples[object_id]) < max_y_samples_per_id
            ):
                object_y_samples[object_id].append(y_value)

            x_value = _int_or_none(extract_object_number(level_object, "2"))
            if x_value is not None:
                if previous_x is not None:
                    delta_x = x_value - previous_x
                    if delta_x > 0 and len(delta_x_samples) < max_delta_samples:
                        delta_x_samples.append(delta_x)
                previous_x = x_value

        produced_chunks = 0
        local_chunk_index = 0
        for start in range(0, len(level_objects), chunk_stride):
            if produced_chunks >= max_chunks_per_level:
                break

            # ── 섹션 경계를 넘는 청크는 건너뜀 (Ch.9: receptive field 경계)
            if objects_cross_portal(level_objects, start, chunk_size):
                continue

            window = level_objects[start : start + chunk_size]
            if len(window) < min_chunk_objects:
                continue

            chunk_counter: Counter[str] = Counter()
            structure_count = 0
            chunk_objects: list[str] = []
            chunk_x_values: list[int] = []
            first_id: str | None = None
            last_id: str | None = None

            for level_object in window:
                object_id = extract_object_id(level_object)
                if not object_id:
                    continue

                x_value = _int_or_none(extract_object_number(level_object, "2"))
                y_value = _int_or_none(extract_object_number(level_object, "3"))
                if x_value is None or y_value is None:
                    continue

                if first_id is None:
                    first_id = object_id
                last_id = object_id

                chunk_counter[object_id] += 1
                if classify(object_id) in (ObjectClass.STRUCTURE, ObjectClass.SPECIAL):
                    structure_count += 1
                chunk_objects.append(level_object)
                chunk_x_values.append(x_value)

            if (
                len(chunk_objects) < min_chunk_objects
                or not chunk_x_values
                or first_id is None
                or last_id is None
            ):
                continue

            # chunk_id는 레벨 내 고유 식별자 — "{level_name}:{local_chunk_index}"
            # 전역 index를 쓰면 레벨 간 transition_counts가 오염됨
            chunk_id = f"{level_name}:{local_chunk_index}"
            local_chunk_index += 1

            chunk_library.append(
                {
                    "chunk_id": chunk_id,
                    "level_name": level_name,
                    "level_desc": level_description[:240],
                    "objects": chunk_objects,
                    "first_id": first_id,
                    "last_id": last_id,
                    "dominant_ids": [
                        obj_id for obj_id, _ in chunk_counter.most_common(12)
                    ],
                    "x_min": min(chunk_x_values),
                    "x_max": max(chunk_x_values),
                    "start_index": start,
                    "is_intro": start <= chunk_stride,
                    "structure_count": structure_count,
                    "object_count": len(chunk_objects),
                }
            )
            produced_chunks += 1

        # 레벨 클래스 분포
        class_counter: Counter[str] = Counter()
        for level_object in level_objects:
            oid = extract_object_id(level_object)
            if oid:
                class_counter[classify(oid).value] += 1
        total_obj = sum(class_counter.values()) or 1

        level_profiles.append(
            {
                "name": level_name,
                "description": level_description[:240],
                "object_count": len(level_objects),
                "top_object_ids": [obj_id for obj_id, _ in level_counter.most_common(80)],
                "class_distribution": {k: round(v / total_obj, 4) for k, v in class_counter.items()},
            }
        )

    chunks_by_level: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunk_library:
        chunks_by_level[str(chunk["level_name"])].append(chunk)

    chunk_transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for chunks in chunks_by_level.values():
        ordered_chunks = sorted(chunks, key=lambda item: int(item.get("start_index", 0)))
        for source, target in zip(ordered_chunks, ordered_chunks[1:]):
            source_id = str(source["chunk_id"])
            target_id = str(target["chunk_id"])
            chunk_transition_counts[source_id][target_id] += 1

    if not delta_x_samples:
        delta_x_samples = [30]

    return {
        "object_prototypes": dict(object_prototypes),
        "object_y_samples": dict(object_y_samples),
        "delta_x_samples": delta_x_samples,
        "level_profiles": level_profiles,
        "chunk_library": chunk_library,
        "chunk_transition_counts": {
            source_id: dict(target_counts)
            for source_id, target_counts in chunk_transition_counts.items()
        },
        "total_sequences": len(sequences),
    }


def train_from_config(config: dict[str, Any]) -> dict[str, Any]:
    dataset_dir = Path(config.get("dataset_dir", "dataset"))
    artifact_path = Path(config.get("artifact_path", "artifacts/model.json"))
    order = int(config.get("markov_order", 2))
    min_objects = _optional_int(config, "min_objects_per_level", None)
    max_objects = _optional_int(config, "max_objects_per_level", None)
    seed = int(config.get("seed", 42))
    log_skipped_files = bool(config.get("log_skipped_files", False))
    max_prototypes_per_id = int(config.get("max_prototypes_per_id", 30))
    max_y_samples_per_id = int(config.get("max_y_samples_per_id", 100))
    max_delta_samples = int(config.get("max_delta_samples", 200000))
    chunk_size = int(config.get("chunk_size", 48))
    chunk_stride = int(config.get("chunk_stride", 24))
    max_chunks_per_level = int(config.get("max_chunks_per_level", 60))
    min_chunk_objects = int(config.get("min_chunk_objects", 12))

    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if chunk_stride < 1:
        raise ValueError("chunk_stride must be >= 1")
    if max_chunks_per_level < 1:
        raise ValueError("max_chunks_per_level must be >= 1")
    if min_chunk_objects < 1:
        raise ValueError("min_chunk_objects must be >= 1")
    if min_chunk_objects > chunk_size:
        raise ValueError("min_chunk_objects cannot be greater than chunk_size")

    set_global_seed(seed)

    load_result = load_dataset_with_report(
        dataset_dir,
        log_skipped_files=log_skipped_files,
    )
    records = load_result.records
    filtered_records = filter_records_by_object_count(
        records,
        min_objects=min_objects,
        max_objects=max_objects,
    )
    if not filtered_records:
        raise ValueError("No records left after object-count filtering.")

    sequences = records_to_token_sequences(filtered_records)
    model = MarkovModel(order=order)
    model.fit(sequences)

    report = evaluate_training_run(
        total_records=load_result.report.files_scanned,
        used_records=len(filtered_records),
        sequences=sequences,
    )
    report["decoded_records"] = load_result.report.loaded_records
    report["skipped_total"] = load_result.report.skipped_total
    report["skipped_missing_k4"] = load_result.report.skipped_missing_k4
    report["skipped_parse_failed"] = load_result.report.skipped_parse_failed
    report["skipped_decode_failed"] = load_result.report.skipped_decode_failed

    template_header = extract_level_header(filtered_records[0].decoded_level_data)
    generation_assets = _build_generation_assets(
        sequences,
        records=filtered_records,
        max_prototypes_per_id=max_prototypes_per_id,
        max_y_samples_per_id=max_y_samples_per_id,
        max_delta_samples=max_delta_samples,
        chunk_size=chunk_size,
        chunk_stride=chunk_stride,
        max_chunks_per_level=max_chunks_per_level,
        min_chunk_objects=min_chunk_objects,
    )

    payload = {
        "model": model.to_dict(),
        "meta": {
            "seed": seed,
            "dataset_dir": str(dataset_dir),
            "report": report,
            "max_prototypes_per_id": max_prototypes_per_id,
            "max_y_samples_per_id": max_y_samples_per_id,
            "max_delta_samples": max_delta_samples,
            "chunk_size": chunk_size,
            "chunk_stride": chunk_stride,
            "max_chunks_per_level": max_chunks_per_level,
            "min_chunk_objects": min_chunk_objects,
        },
        "template_tags": _serialize_tags(filtered_records[0].document.tags),
        "template_level_header": template_header,
        "token_frequency": object_token_frequencies(sequences, top_n=100),
        "generation_assets": generation_assets,
        "reference_stats": build_reference_stats_from_records(filtered_records),
        "audio_conditioned_model_spec": build_audio_conditioned_model_spec().to_dict(),
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "artifact_path": str(artifact_path),
        **report,
    }
