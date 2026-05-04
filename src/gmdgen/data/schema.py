# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

TagEntry = tuple[str, str]
TagMap = dict[str, TagEntry]


@dataclass(slots=True)
class GMDDocument:
    path: Path
    raw_text: str
    tags: TagMap


@dataclass(slots=True)
class GMDRecord:
    document: GMDDocument
    decoded_level_data: str


@dataclass(slots=True)
class SectionInfo:
    """One contiguous section of a GD level, bounded by portals.

    Goodfellow Ch.9 §9.2 — equivalent to a 'receptive field boundary':
    objects inside a section share the same speed/gamemode context,
    so they form a natural, coherent local pattern for the model.
    """

    section_id: int
    start_object_index: int
    end_object_index: int        # exclusive
    gamemode: str = "cube"       # cube/ship/ball/ufo/wave/robot/spider/swing
    speed: str = "normal"        # half/normal/double/triple/quadruple
    object_count: int = 0


@dataclass
class ObjectFeature:
    """Feature vector for a single GD object.

    Goodfellow Ch.1 "representation learning" + Ch.15 §15.3
    "Disentangling Factors of Variation":
      - identity    : what the object *is*  (id, class)
      - position    : where it *is*         (x, y, dx from previous)
      - spatial band: vertical zone         (y_bucket)
      - relation    : group/trigger links   (group_ids, trigger_target)
      - context     : section the object    (section_id, speed, gamemode)
    """

    object_id: str
    object_class: str                       # "structure"/"decoration"/"trigger"/"portal"/"special"/"unknown"
    x: float
    y: float
    dx: float = 0.0                         # x − x_{i−1}  (0 for first object)
    y_bucket: int = 0                       # vertical zone index (0–7)
    section_id: int = 0
    group_ids: list[int] = field(default_factory=list)
    trigger_target: Optional[int] = None
    raw: str = ""                           # original GD object string


@dataclass(slots=True)
class DatasetLoadReport:
    files_scanned: int
    loaded_records: int
    skipped_missing_k4: int
    skipped_parse_failed: int
    skipped_decode_failed: int

    @property
    def skipped_total(self) -> int:
        return (
            self.skipped_missing_k4
            + self.skipped_parse_failed
            + self.skipped_decode_failed
        )


@dataclass(slots=True)
class DatasetLoadResult:
    records: list[GMDRecord]
    report: DatasetLoadReport
