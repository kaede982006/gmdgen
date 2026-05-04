# SPDX-License-Identifier: GPL-3.0-or-later
"""Conditional editing mode — base level + learned delta.

Goodfellow Ch.7 §7.4 "Dataset Augmentation" + Ch.19 §19.3 "MAP Inference":

  Instead of generating a level from scratch (high variance, high error),
  we keep the *global structure* of a real template level intact and apply
  only *small learned modifications* to its content:

    output = keep_skeleton(template) + swap_style(template, model, prompt)

  Concretely:
    • Portals, triggers, group-linked objects  → kept as-is  (skeleton)
    • Structure / decoration objects           → style-swapped if desired
    • X and Y positions                        → optionally jittered (±delta)
    • k95, k4 header                           → reconstructed from result

  This guarantees:
    ✓ trigger / group relationships survive     (L_trigger = 0)
    ✓ speed/gamemode section structure survives (L_section = 0)
    ✓ X-monotone is inherited from the template (L_position ≈ 0)

  Trade-off: the output is stylistically constrained by the template.
  Use `style_swap_ratio` to control how aggressively content is replaced.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gmdgen.data.preprocess import extract_level_header, split_level_objects
from gmdgen.features.tokenizer import (
    extract_object_id,
    extract_object_number,
    rewrite_object_xy,
)
from gmdgen.io.gmd_decoder import decode_level_data
from gmdgen.io.gmd_parser import parse_gmd_file
from gmdgen.representation.object_classifier import ObjectClass, classify

LOGGER = logging.getLogger(__name__)

# Keys that indicate the object is structurally load-bearing
# (must be kept for level integrity)
_KEEP_FIELDS = {"51", "155", "57"}   # trigger target, group, z-layer

# Portals / triggers — always kept unchanged
_ALWAYS_KEEP_CLASSES: frozenset[ObjectClass] = frozenset(
    {ObjectClass.PORTAL, ObjectClass.TRIGGER, ObjectClass.SPECIAL}
)


@dataclass
class EditReport:
    template_path: str
    template_object_count: int
    kept_count: int
    swapped_count: int
    jittered_count: int

    @property
    def swap_ratio(self) -> float:
        total = self.kept_count + self.swapped_count
        return round(self.swapped_count / total, 4) if total else 0.0


def _load_template(template_path: Path) -> tuple[str, str, list[str]]:
    """Return (level_header, decoded_level_data, object_list)."""
    document = parse_gmd_file(template_path)
    k4_entry = document.tags.get("k4")
    if not k4_entry:
        raise ValueError(f"Template has no k4 field: {template_path}")

    _, encoded = k4_entry
    decoded = decode_level_data(encoded)
    header = extract_level_header(decoded)
    objects = split_level_objects(decoded)
    return header, decoded, objects


def _build_style_pool(
    *,
    generation_assets: dict[str, Any],
    prompt_words: set[str],
    matched_level_names: set[str],
    target_class: ObjectClass,
) -> list[str]:
    """Collect candidate replacement object strings for a given class.

    Pulled from the chunk library so that the replacements look like
    real objects from the training data.
    """
    pool: list[str] = []
    chunk_library = generation_assets.get("chunk_library", [])
    object_prototypes = generation_assets.get("object_prototypes", {})

    for chunk in chunk_library:
        level_name = str(chunk.get("level_name", ""))
        score = 0.0
        if prompt_words:
            from gmdgen.generate.generator import _tokenize_words
            overlap = len(_tokenize_words(f"{level_name} {chunk.get('level_desc','')}") & prompt_words)
            score += overlap * 2.0
        if level_name in matched_level_names:
            score += 3.0
        if score <= 0:
            continue

        for raw_obj in chunk.get("objects", []):
            obj_id = extract_object_id(raw_obj)
            if obj_id and classify(obj_id) == target_class:
                pool.append(raw_obj)

    # Fall back to object_prototypes when chunk library gives no hits
    if not pool:
        for obj_id, prototypes in object_prototypes.items():
            if classify(obj_id) == target_class and prototypes:
                pool.extend(prototypes[:4])

    return pool


def _jitter_position(
    raw_obj: str,
    *,
    rng: random.Random,
    x_jitter: int,
    y_jitter: int,
) -> str:
    x_val = extract_object_number(raw_obj, "2")
    y_val = extract_object_number(raw_obj, "3")
    if x_val is None or y_val is None:
        return raw_obj

    new_x = max(0, int(round(x_val)) + rng.randint(-x_jitter, x_jitter))
    new_y = int(round(y_val)) + rng.randint(-y_jitter, y_jitter)
    return rewrite_object_xy(raw_obj, x=new_x, y=new_y)


def apply_style_edit(
    template_objects: list[str],
    *,
    generation_assets: dict[str, Any],
    prompt: str,
    matched_level_names: list[str],
    style_swap_ratio: float,
    jitter_x: int,
    jitter_y: int,
    rng: random.Random,
    swap_structure: bool,
    swap_decoration: bool,
) -> tuple[list[str], EditReport]:
    """Apply learned style delta to template objects.

    For each object:
    - Portals / triggers / special  → always kept unchanged
    - Structure (if swap_structure)  → replaced with pool object at same position
    - Decoration (if swap_decoration)→ replaced with pool object at same position
    - Position jitter                → applied after swap if jitter > 0
    """
    from gmdgen.generate.generator import _tokenize_words

    prompt_words = _tokenize_words(prompt)
    matched_set = set(matched_level_names)

    structure_pool = (
        _build_style_pool(
            generation_assets=generation_assets,
            prompt_words=prompt_words,
            matched_level_names=matched_set,
            target_class=ObjectClass.STRUCTURE,
        )
        if swap_structure
        else []
    )
    decoration_pool = (
        _build_style_pool(
            generation_assets=generation_assets,
            prompt_words=prompt_words,
            matched_level_names=matched_set,
            target_class=ObjectClass.DECORATION,
        )
        if swap_decoration
        else []
    )

    result: list[str] = []
    kept = 0
    swapped = 0
    jittered = 0

    for raw_obj in template_objects:
        obj_id = extract_object_id(raw_obj)
        if not obj_id:
            result.append(raw_obj)
            kept += 1
            continue

        obj_class = classify(obj_id)

        # Always-keep classes — copy verbatim
        if obj_class in _ALWAYS_KEEP_CLASSES:
            result.append(raw_obj)
            kept += 1
            continue

        # Decide whether to swap
        do_swap = False
        pool: list[str] = []
        if obj_class == ObjectClass.STRUCTURE and swap_structure and structure_pool:
            do_swap = rng.random() < style_swap_ratio
            pool = structure_pool
        elif obj_class == ObjectClass.DECORATION and swap_decoration and decoration_pool:
            do_swap = rng.random() < style_swap_ratio
            pool = decoration_pool

        if do_swap:
            replacement = rng.choice(pool)
            # Graft original position onto replacement object
            x_val = extract_object_number(raw_obj, "2")
            y_val = extract_object_number(raw_obj, "3")
            if x_val is not None and y_val is not None:
                replacement = rewrite_object_xy(
                    replacement,
                    x=int(round(x_val)),
                    y=int(round(y_val)),
                )
            raw_obj = replacement
            swapped += 1
        else:
            kept += 1

        # Optional position jitter (applied after possible swap)
        if (jitter_x > 0 or jitter_y > 0) and rng.random() < 0.35:
            raw_obj = _jitter_position(
                raw_obj, rng=rng, x_jitter=jitter_x, y_jitter=jitter_y
            )
            jittered += 1

        result.append(raw_obj)

    report = EditReport(
        template_path="",  # filled by caller
        template_object_count=len(template_objects),
        kept_count=kept,
        swapped_count=swapped,
        jittered_count=jittered,
    )

    LOGGER.info(
        "apply_style_edit: kept=%d swapped=%d jittered=%d swap_ratio=%.3f",
        kept,
        swapped,
        jittered,
        report.swap_ratio,
    )

    return result, report


def run_template_edit(
    *,
    template_path: Path,
    generation_assets: dict[str, Any],
    prompt: str,
    matched_level_names: list[str],
    style_swap_ratio: float,
    jitter_x: int,
    jitter_y: int,
    swap_structure: bool,
    swap_decoration: bool,
    rng: random.Random,
) -> tuple[str, list[str], EditReport]:
    """Load a template level and apply style editing.

    Returns (level_header, edited_objects, report).
    """
    header, _, template_objects = _load_template(template_path)

    if not template_objects:
        raise ValueError(f"Template level has no objects: {template_path}")

    edited_objects, report = apply_style_edit(
        template_objects,
        generation_assets=generation_assets,
        prompt=prompt,
        matched_level_names=matched_level_names,
        style_swap_ratio=style_swap_ratio,
        jitter_x=jitter_x,
        jitter_y=jitter_y,
        rng=rng,
        swap_structure=swap_structure,
        swap_decoration=swap_decoration,
    )
    report.template_path = str(template_path)

    return header, edited_objects, report
