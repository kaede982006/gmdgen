# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import random
import math
from dataclasses import dataclass, field
from typing import Any

from gmdgen.gd.plans import ObjectPlan, SectionPlan, TriggerPlan
from gmdgen.gd.time_mapping import SpeedState, pos_for_time_like_gd, time_for_pos_like_gd
from gmdgen.representation.object_classifier import ObjectClass, classify
from gmdgen.features.tokenizer import extract_object_id, extract_object_number


# Safe, diverse default decoration palette. These IDs are non-gameplay editor-safe
# blocks/decorations that the repairer will not flag as hazards. Replaces the
# previous degenerate ["211"] fallback that collapsed object_diversity_score.
_SAFE_DECORATION_PALETTE: list[str] = [
    "211", "259", "266", "273", "658", "503", "504", "1734", "1736",
    "1735", "1737", "1764", "1765", "1766", "1767", "1768", "1769",
    "1770", "1771", "1772", "1773", "1774", "1775", "1776", "1777",
    "1778", "1779", "1780", "1781", "1782", "1783",
]
_SAFE_BACKGROUND_PALETTE: list[str] = [
    "211", "503", "504", "1734", "1736",
]
_SAFE_STRUCTURAL_PALETTE: list[str] = [
    "1", "2", "3", "5", "6", "7",
]
# Per-section motif palettes used to rotate object IDs across sections so two
# adjacent sections do not produce identical decoration tuples.
_SECTION_MOTIF_FAMILIES: list[list[str]] = [
    ["211", "503", "504"],
    ["259", "1734", "1736"],
    ["266", "273", "658"],
    ["503", "1734", "211"],
]


def _palette_for_section(section_index: int, base: list[str]) -> list[str]:
    """Rotate base palette so each section gets a slightly different mix.

    The motif family rotates by section_index; when base is empty, we use
    the safe palette rotated so adjacent sections start from different IDs.
    """
    family = _SECTION_MOTIF_FAMILIES[section_index % len(_SECTION_MOTIF_FAMILIES)]
    if not base:
        # Even when there's no learned/style palette, rotate the safe palette
        # so adjacent sections produce different leading IDs.
        offset = section_index * 3 % len(_SAFE_DECORATION_PALETTE)
        rotated = _SAFE_DECORATION_PALETTE[offset:] + _SAFE_DECORATION_PALETTE[:offset]
        merged: list[str] = list(family)
        seen: set[str] = set(merged)
        for item in rotated:
            if item not in seen:
                seen.add(item)
                merged.append(item)
        return merged
    merged = []
    seen = set()
    for item in list(base) + family:
        s = str(item)
        if s and s not in seen:
            seen.add(s)
            merged.append(s)
    return merged or list(_SAFE_DECORATION_PALETTE)


@dataclass(slots=True)
class MaterializationConfig:
    object_multiplier: float = 1.0
    target_object_count: int | None = None
    min_object_count: int = 100
    max_object_count: int = 40000
    detail_density: float = 0.5
    decoration_density: float = 0.5
    gameplay_density: float = 0.5
    sync_accent_density: float = 0.5
    section_object_floor: int = 10
    section_object_ceiling: int = 10000
    fast_materialization: bool = True
    seed: int = 42


class SectionObjectMaterializer:
    def __init__(self, config: MaterializationConfig):
        self.config = config

    def materialize_section(
        self,
        section: SectionPlan,
        *,
        audio_features: Any = None,
        motif_library: list[dict[str, Any]] | None = None,
        style_profile: dict[str, Any] | None = None,
        object_budget: int = 2000,
        speed_objects: list[Any] | None = None,
        start_speed: SpeedState = SpeedState.NORMAL,
        song_offset: float = 0.0,
        section_index: int = 0,
    ) -> list[ObjectPlan]:
        """Materialize objects for a single section.

        After collection, all objects are stable-sorted by x so the section
        is x-monotonic by construction. This eliminates the need for
        downstream x_mono repair.
        """
        objects: list[ObjectPlan] = []

        section_width = section.end_x - section.start_x
        if section_width <= 0:
            return []

        # Deterministic generation per section
        section_rng = random.Random(self.config.seed + int(section.start_x))

        # 1. Beat-synchronized gameplay objects (skeleton)
        if audio_features and hasattr(audio_features, "beat_times"):
            beat_objects = self._materialize_beat_sync_objects(
                section,
                audio_features.beat_times,
                section_rng,
                speed_objects=speed_objects,  # type: ignore
                start_speed=start_speed,
                song_offset=song_offset,
            )
            objects.extend(beat_objects)

        # 2. Motif expansion
        if motif_library:
            motif_objects = self._materialize_motif_objects(
                section,
                motif_library,
                section_rng,
                object_budget - len(objects)
            )
            objects.extend(motif_objects)

        # 3. Onset-synchronized accents
        if audio_features and hasattr(audio_features, "onset_times"):
            accent_objects = self._materialize_onset_accents(
                section,
                audio_features.onset_times,
                section_rng,
                speed_objects=speed_objects,  # type: ignore
                start_speed=start_speed,
                song_offset=song_offset,
            )
            objects.extend(accent_objects)

        # 4. Fill with decorations to reach target if needed (x-monotonic walk)
        target_count = self._calculate_target_count(section, section_width, object_budget)
        if len(objects) < target_count:
            fill_objects = self._materialize_fill_decorations(
                section,
                target_count - len(objects),
                style_profile,
                section_rng,
                section_index=section_index,
            )
            objects.extend(fill_objects)

        # Clamp x to section range and stable-sort by x for monotonic output.
        for obj in objects:
            if obj.x < section.start_x:
                obj.x = section.start_x
            elif obj.x > section.end_x:
                obj.x = section.end_x
        objects.sort(key=lambda o: o.x)
        return objects[:object_budget]

    def _calculate_target_count(self, section: SectionPlan, width: float, budget: int) -> int:
        # Base count: 1 object per 20 units at 1.0 density
        base = (width / 20.0) * section.density_target
        target = int(base * self.config.object_multiplier)
        
        if self.config.target_object_count:
            # If global target is set, we'll handle it in materialize_level_plans
            pass
            
        return max(self.config.section_object_floor, min(target, self.config.section_object_ceiling, budget))

    def _materialize_beat_sync_objects(
        self,
        section: SectionPlan,
        beat_times: list[float],
        rng: random.Random,
        speed_objects: list[Any],
        start_speed: SpeedState,
        song_offset: float,
    ) -> list[ObjectPlan]:
        objects = []
        # Filter beats in section
        section_beats = [t for t in beat_times if section.start_time <= t < section.end_time]
        
        for t in section_beats:
            if rng.random() > section.density_target * self.config.gameplay_density:
                continue
                
            x = pos_for_time_like_gd(t, speed_objects, start_speed, song_offset)
            
            # Simple jump structure or orb
            if rng.random() > 0.5:
                # Orb
                objects.append(ObjectPlan(
                    object_id=rng.choice(["36", "141", "1332"]), # Yellow, Red, Green orbs
                    x=x,
                    y=rng.randint(60, 150),
                    role="beat_orb",
                    beat_aligned_time=t
                ))
            else:
                # Basic block/platform
                objects.append(ObjectPlan(
                    object_id="1",
                    x=x,
                    y=30 * rng.randint(1, 3),
                    role="beat_structure",
                    beat_aligned_time=t
                ))
        return objects

    def _materialize_motif_objects(
        self,
        section: SectionPlan,
        motif_library: list[dict[str, Any]],
        rng: random.Random,
        remaining_budget: int
    ) -> list[ObjectPlan]:
        objects = []  # type: ignore
        if remaining_budget <= 0:
            return []
            
        # Number of motifs to place based on intensity
        num_motifs = int((section.end_x - section.start_x) / 500 * section.decoration_intensity * 2)
        num_motifs = max(1, min(num_motifs, 10))
        
        visible_chunks = [
            chunk for chunk in motif_library
            if isinstance(chunk.get("objects"), list) and chunk.get("objects")
        ]
        if not visible_chunks:
            return []

        for _ in range(num_motifs):
            if len(objects) >= remaining_budget:
                break
            chunk = rng.choice(visible_chunks)
            chunk_objs = chunk.get("objects", [])
            if not chunk_objs:
                continue
                
            # Random position for the motif anchor
            anchor_x = rng.uniform(section.start_x, section.end_x - 300)
            anchor_y = rng.uniform(30, 150)
            
            # Extract objects from chunk and offset them
            first_x = None
            for raw_obj in chunk_objs:
                obj_id = extract_object_id(str(raw_obj))
                ox = extract_object_number(str(raw_obj), "2")
                oy = extract_object_number(str(raw_obj), "3")
                if obj_id is None or ox is None or oy is None:
                    continue
                
                if first_x is None:
                    first_x = ox
                
                objects.append(ObjectPlan(
                    object_id=obj_id,
                    x=anchor_x + (ox - first_x),
                    y=anchor_y + oy,
                    role="materialized_motif",
                    safety_flags={"source": "motif_expander"}
                ))
                if len(objects) >= remaining_budget:
                    break
        return objects

    def _materialize_onset_accents(
        self,
        section: SectionPlan,
        onset_times: list[float],
        rng: random.Random,
        speed_objects: list[Any],
        start_speed: SpeedState,
        song_offset: float,
    ) -> list[ObjectPlan]:
        objects = []
        section_onsets = [t for t in onset_times if section.start_time <= t < section.end_time]
        
        for t in section_onsets:
            if rng.random() > section.trigger_intensity * self.config.sync_accent_density:
                continue
                
            x = pos_for_time_like_gd(t, speed_objects, start_speed, song_offset)
            # Add a visual decoration on onset
            objects.append(ObjectPlan(
                object_id="211", # Simple pulse-like deco
                x=x,
                y=rng.randint(200, 300),
                role="sync_accent",
                beat_aligned_time=t
            ))
        return objects

    def _materialize_fill_decorations(
        self,
        section: SectionPlan,
        count: int,
        style_profile: dict[str, Any] | None,
        rng: random.Random,
        section_index: int = 0,
    ) -> list[ObjectPlan]:
        """Generate fill decorations using a diverse, x-monotonic schedule.

        Replaces the prior O(n) random-x scatter approach which forced the
        repairer to do massive x_mono fixes.  Now we walk a monotonic cursor
        across the section, rotate IDs through a safe palette per-section so
        adjacent sections don't produce identical tuples, and use vertical
        bands keyed off role so decoration never sits on the gameplay path.
        """
        objects: list[ObjectPlan] = []
        if count <= 0 or section.end_x <= section.start_x:
            return objects

        # Build palette: prefer style_profile ids, fall back to the safe palette.
        base_ids: list[str] = []
        if style_profile:
            ids_by_class = style_profile.get("ids_by_class") or {}
            base_ids = list(ids_by_class.get(ObjectClass.DECORATION.value, []))
        if not base_ids:
            base_ids = list(_SAFE_DECORATION_PALETTE)
        palette = _palette_for_section(section_index, base_ids)

        # Y-bands keep decoration above gameplay corridor (y=0..150 reserved
        # for gameplay) so decorations never sit in the player path.
        deco_y_band = (180, 380)
        bg_y_band = (300, 540)

        width = section.end_x - section.start_x
        # Step is the average spacing; jitter keeps it deterministic but visually varied.
        step = max(8.0, width / max(1, count))
        x = section.start_x
        seen_tuples: set[tuple[str, int, int]] = set()

        for i in range(count):
            # Monotonic advance with deterministic jitter.
            jitter = (rng.random() - 0.5) * step * 0.4
            x = min(section.end_x - 1.0, x + step + jitter)

            obj_id = palette[(i + section_index) % len(palette)]

            # Alternate decoration / background-detail role to add structural variety.
            if i % 5 == 4:
                role = "background_detail"
                y_min, y_max = bg_y_band
            else:
                role = "fill_decoration"
                y_min, y_max = deco_y_band
            y = rng.randint(y_min, y_max)

            # Suppress exact duplicate tuples to keep diversity high.
            key = (obj_id, int(x // 30), y // 15)
            if key in seen_tuples:
                obj_id = palette[(i + section_index + 3) % len(palette)]
                key = (obj_id, int(x // 30), y // 15)
            seen_tuples.add(key)

            objects.append(ObjectPlan(
                object_id=str(obj_id),
                x=x,
                y=y,
                role=role,
                editor_layer=2,
            ))
        return objects


def materialize_level_plans(
    section_plans: list[SectionPlan],
    *,
    config: MaterializationConfig,
    audio_features: Any = None,
    motif_library: list[dict[str, Any]] | None = None,
    style_profile: dict[str, Any] | None = None,
    total_object_budget: int = 10000,
    speed_objects: list[Any] = None,  # type: ignore
    start_speed: SpeedState = SpeedState.NORMAL,
    song_offset: float = 0.0,
) -> list[ObjectPlan]:
    """Materialize section plans into a globally x-monotonic ObjectPlan list.

    Sections are processed in start_x order, each section internally is sorted
    by x, and finally a single stable global sort ensures the output is
    x-monotonic by construction. This eliminates downstream x_mono repair
    work that previously caused massive repair_loss.
    """
    materializer = SectionObjectMaterializer(config)
    all_objects: list[ObjectPlan] = []

    total_width = sum(s.end_x - s.start_x for s in section_plans) or 1.0

    # If target_object_count is set, we adjust total_object_budget
    actual_budget = total_object_budget
    if config.target_object_count:
        actual_budget = min(config.target_object_count, config.max_object_count)

    # Sort sections in start_x order so concatenation respects level order.
    sorted_sections = sorted(enumerate(section_plans), key=lambda item: item[1].start_x)

    for section_index, section in sorted_sections:
        section_width = section.end_x - section.start_x
        section_budget = int(actual_budget * (section_width / total_width))

        section_objects = materializer.materialize_section(
            section,
            audio_features=audio_features,
            motif_library=motif_library,
            style_profile=style_profile,
            object_budget=section_budget,
            speed_objects=speed_objects,
            start_speed=start_speed,
            song_offset=song_offset,
            section_index=section_index,
        )
        all_objects.extend(section_objects)

    # Final safety stable-sort across section boundaries.
    all_objects.sort(key=lambda o: o.x)
    return all_objects
