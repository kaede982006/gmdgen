# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field

from gmdgen.data.schema import GMDRecord

# 게임모드 포탈 오브젝트 ID
_GAMEMODE_PORTAL_IDS: frozenset[str] = frozenset(
    {"12", "13", "47", "111", "660", "745", "1331"}
)

# 속도 포탈 오브젝트 ID → speed label
_SPEED_PORTAL_IDS: dict[str, str] = {
    "200": "half",
    "201": "normal",
    "202": "double",
    "203": "triple",
    "1334": "quadruple",
}


@dataclass(slots=True)
class SectionBoundary:
    start_object_index: int
    gamemode: str = "cube"
    speed: str = "normal"


def split_level_sections(level_data: str) -> list[str]:
    return [chunk.strip() for chunk in level_data.split(";") if chunk.strip()]


def _looks_like_header(section: str) -> bool:
    if section.startswith("kS") or section.startswith("kA"):
        return True
    if ",kA" in section or ",kS" in section:
        return True
    return False


def split_header_and_objects(level_data: str) -> tuple[str, list[str]]:
    sections = split_level_sections(level_data)
    if not sections:
        return "", []

    first = sections[0]
    if _looks_like_header(first):
        return first, [chunk for chunk in sections[1:] if chunk.startswith("1,")]

    return "", [chunk for chunk in sections if chunk.startswith("1,")]


def extract_level_header(level_data: str) -> str:
    header, _ = split_header_and_objects(level_data)
    return header


def split_level_objects(level_data: str) -> list[str]:
    _, objects = split_header_and_objects(level_data)
    return objects


def detect_section_boundaries(objects: list[str]) -> list[SectionBoundary]:
    from gmdgen.features.tokenizer import extract_object_id

    boundaries: list[SectionBoundary] = [SectionBoundary(start_object_index=0)]
    current_gamemode = "cube"
    current_speed = "normal"

    for idx, obj in enumerate(objects):
        obj_id = extract_object_id(obj)
        if obj_id is None:
            continue

        if obj_id in _GAMEMODE_PORTAL_IDS:
            gamemode_map = {
                "12": "ship", "13": "ball", "47": "ufo",
                "111": "wave", "660": "robot", "745": "spider", "1331": "swing",
            }
            current_gamemode = gamemode_map.get(obj_id, current_gamemode)
            if idx > 0:
                boundaries.append(
                    SectionBoundary(
                        start_object_index=idx,
                        gamemode=current_gamemode,
                        speed=current_speed,
                    )
                )

        elif obj_id in _SPEED_PORTAL_IDS:
            current_speed = _SPEED_PORTAL_IDS[obj_id]
            if idx > 0:
                boundaries.append(
                    SectionBoundary(
                        start_object_index=idx,
                        gamemode=current_gamemode,
                        speed=current_speed,
                    )
                )

    return boundaries


def objects_cross_portal(
    objects: list[str],
    start: int,
    size: int,
) -> bool:
    from gmdgen.features.tokenizer import extract_object_id

    portal_ids = _GAMEMODE_PORTAL_IDS | set(_SPEED_PORTAL_IDS.keys())
    for idx in range(max(0, start), min(len(objects), start + size)):
        obj_id = extract_object_id(objects[idx])
        if obj_id in portal_ids:
            return True
    return False


def filter_records_by_object_count(
    records: list[GMDRecord],
    *,
    min_objects: int | None,
    max_objects: int | None,
) -> list[GMDRecord]:
    filtered: list[GMDRecord] = []
    for record in records:
        object_count = len(split_level_objects(record.decoded_level_data))
        if min_objects is not None and object_count < min_objects:
            continue
        if max_objects is not None and object_count > max_objects:
            continue
        filtered.append(record)
    return filtered
