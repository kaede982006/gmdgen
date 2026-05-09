# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 kaede982006
"""Constrained nucleus-sampling decoder for the GMD language model.

We produce object id / dx-bin / y-bin per step (Foundations of LLMs §5.1.3
— decoding algorithms). To keep editor-validity we apply two masks:

  * id mask: must come from the trained id vocabulary (UNK collapses).
  * dx mask: must produce a strictly non-negative bin so x is monotonic.

After sampling, we *materialize* the predicted tokens back into
:class:`ExpandedObject` instances usable by the existing IO/.gmd writer.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from gmdgen.utils.device import get_best_device, get_device_info
from gmdgen.generate.expand import ExpandedObject
from gmdgen.ml.architectures import GMDLanguageModel
from gmdgen.ml.tokens import (
    BOS_ID,
    DEFAULT_MODE_IDX,
    DEFAULT_SPEED_IDX,
    DX_BINS,
    EOS_ID,
    NUM_SPECIAL,
    PAD_ID,
    UNK_ID,
    Y_BINS,
    IdVocab,
    class_idx_for_object_id,
    dx_bucket,
    mode_portal_object_id,
    update_running_state,
    y_bucket,
)
from gmdgen.representation.object_classifier import ObjectClass, classify


@dataclass(slots=True)
class SamplingConfig:
    max_objects: int = 600
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 40
    seed: int = 0
    candidates: int = 3
    prompt: str = ""
    add_ground_rail: bool = True
    min_dx: float = 15.0
    rail_step: float = 40.0
    repetition_penalty: float = 0.8


@dataclass(slots=True)
class _SampleCandidate:
    objects: list[ExpandedObject]
    score: float
    diagnostics: dict[str, Any]


def _slot_to_object_id(vocab: IdVocab) -> dict[int, str]:
    inv: dict[int, str] = {}
    for oid, slot in vocab.id_to_slot.items():
        inv[slot] = oid
    return inv


def _filter_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    forbidden: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply temperature, top-k, top-p, and optional forbidden mask."""
    logits = logits / max(1e-6, temperature)
    if forbidden is not None:
        logits = logits.masked_fill(forbidden, float("-inf"))
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k=k).values[..., -1, None]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cum > top_p
        # always keep the most probable token
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = torch.zeros_like(logits).scatter(-1, sorted_idx, sorted_logits)
    return logits


def _bin_center_dx(idx: int) -> float:
    if idx <= 0:
        return DX_BINS[0]
    if idx >= len(DX_BINS) - 1:
        return DX_BINS[-2]
    return 0.5 * (DX_BINS[idx] + DX_BINS[idx + 1])


def _bin_center_y(idx: int) -> float:
    if idx <= 0:
        return Y_BINS[0]
    if idx >= len(Y_BINS) - 1:
        return Y_BINS[-2]
    return 0.5 * (Y_BINS[idx] + Y_BINS[idx + 1])


def _role_for_object_id(object_id: str) -> str:
    cls = classify(object_id)
    if cls == ObjectClass.DECORATION:
        return "decoration"
    if cls == ObjectClass.TRIGGER:
        return "trigger"
    if cls == ObjectClass.PORTAL:
        return "portal"
    if cls == ObjectClass.SPECIAL:
        return "gameplay"
    return "structural"


def _terrain_y(index: int, section_idx: int) -> float:
    terrain = (90.0, 105.0, 120.0, 105.0, 90.0, 120.0)
    return terrain[(index + section_idx * 2) % len(terrain)]


def _initial_mode_from_prompt(prompt: str) -> int:
    lowered = prompt.lower()
    for idx, name in enumerate(("cube", "ship", "ball", "ufo", "wave", "robot", "spider")):
        if name in lowered:
            return idx
    return DEFAULT_MODE_IDX


def _append_history(
    *,
    object_id: str,
    id_sample: int,
    step_dx: float,
    cur_y: float,
    cur_section: int,
    ids: list[int],
    cls: list[int],
    dx: list[int],
    y: list[int],
    mode: list[int],
    speed: list[int],
    section: list[int],
) -> None:
    next_mode, next_speed = update_running_state(
        object_id,
        mode_idx=mode[-1],
        speed_idx=speed[-1],
    )
    ids.append(id_sample)
    cls.append(class_idx_for_object_id(object_id))
    dx.append(dx_bucket(step_dx))
    y.append(y_bucket(cur_y))
    mode.append(next_mode)
    speed.append(next_speed)
    section.append(min(cur_section, 31))


def _bias_id_logits(
    logits: torch.Tensor,
    *,
    inv_id: dict[int, str],
    objects: list[ExpandedObject],
    cfg: SamplingConfig,
) -> torch.Tensor:
    """Steer away from degenerate repetition while preserving model ranking."""
    if not objects:
        return logits

    logits = logits.clone()
    recent = Counter(o.object_id for o in objects[-16:])
    all_counts = Counter(o.object_id for o in objects)
    class_counts = Counter(_role_for_object_id(o.object_id) for o in objects)
    total = max(1, len(objects))
    decor_ratio = class_counts["decoration"] / total
    structural_ratio = class_counts["structural"] / total
    special_ratio = (class_counts["gameplay"] + class_counts["portal"]) / total

    for slot, object_id in inv_id.items():
        penalty = 0.0
        if recent[object_id] > 0:
            penalty += cfg.repetition_penalty * recent[object_id]
        if all_counts[object_id] / total > 0.20:
            penalty += 0.6

        role = _role_for_object_id(object_id)
        if role == "decoration" and decor_ratio > 0.45:
            penalty += 0.5
        elif role == "structural" and structural_ratio > 0.72:
            penalty += 0.35
        elif role in {"portal", "gameplay"} and special_ratio < 0.08:
            penalty -= 0.25
        elif role == "trigger" and class_counts["trigger"] < 2 and total > 24:
            penalty -= 0.15

        if penalty:
            logits[slot] -= penalty
    return logits


def _bias_small_vocab_logits(logits: torch.Tensor, history: list[int]) -> torch.Tensor:
    """Avoid single-bin collapse for dx/y heads in tiny checkpoints."""
    if len(history) < 6:
        return logits
    logits = logits.clone()
    recent = Counter(history[-12:])
    for idx, count in recent.items():
        if count >= 4 and 0 <= idx < logits.size(-1):
            logits[idx] -= 0.55 + 0.08 * count
    if logits.size(-1) > 1:
        logits[0] -= 0.35
    return logits


def _candidate_score(objects: list[ExpandedObject], *, sections: int) -> tuple[float, dict[str, Any]]:
    if not objects:
        return -1.0, {"raw_object_count": 0}

    total = len(objects)
    ids = [o.object_id for o in objects]
    y_bins = [round(o.y / 30.0) for o in objects]
    roles = Counter(_role_for_object_id(o.object_id) for o in objects)
    id_counts = Counter(ids)
    y_counts = Counter(y_bins)

    unique_score = min(1.0, len(id_counts) / 18.0)
    y_entropy_proxy = 1.0 - (max(y_counts.values()) / total)
    y_spread = min(1.0, (max(o.y for o in objects) - min(o.y for o in objects)) / 360.0)
    repeat_penalty = max(id_counts.values()) / total
    ground_flat_ratio = sum(1 for o in objects if abs(o.y - 105.0) <= 5.0) / total
    class_balance = 1.0 - abs((roles["structural"] / total) - 0.45)
    gameplay_presence = min(1.0, (roles["gameplay"] + roles["portal"] + roles["trigger"]) / max(1.0, total * 0.12))

    sec_counts = Counter(o.section_id for o in objects)
    expected_sections = max(1, sections)
    section_coverage = min(1.0, len(sec_counts) / expected_sections)

    score = (
        0.24 * unique_score
        + 0.20 * y_entropy_proxy
        + 0.18 * y_spread
        + 0.16 * class_balance
        + 0.14 * gameplay_presence
        + 0.08 * section_coverage
        - 0.20 * repeat_penalty
        - 0.12 * ground_flat_ratio
    )
    diagnostics = {
        "raw_object_count": total,
        "candidate_score": round(score, 6),
        "unique_object_ids": len(id_counts),
        "max_repeated_id_ratio": round(repeat_penalty, 6),
        "y_entropy_proxy": round(y_entropy_proxy, 6),
        "y_spread": round(y_spread, 6),
        "ground_flat_ratio": round(ground_flat_ratio, 6),
        "role_counts": dict(roles),
    }
    return score, diagnostics


def _add_support_rail(
    objects: list[ExpandedObject],
    *,
    cfg: SamplingConfig,
) -> tuple[list[ExpandedObject], int]:
    if not objects or not cfg.add_ground_rail:
        return objects, 0

    x_min = min(o.x for o in objects)
    x_max = max(o.x for o in objects)
    min_bin = int(max(0.0, x_min - 30.0) // 30.0)
    max_bin = int((x_max + 30.0) // 30.0)
    total_bins = max(1, max_bin - min_bin + 1)
    filled_bins = {
        int(float(o.x) // 30.0)
        for o in objects
        if abs(float(o.y) - 105.0) <= 35.0
    }
    target_bins = int(total_bins * 0.75)
    if target_bins < total_bins * 0.75:
        target_bins += 1
    if len(filled_bins) >= target_bins:
        return objects, 0

    rail: list[ExpandedObject] = []
    i = 0
    rail_ids = ("1", "2", "3", "4", "5", "6")
    rail_y = (90.0, 105.0, 120.0, 105.0)
    for bin_idx in range(min_bin, max_bin + 1):
        if len(filled_bins) >= target_bins:
            break
        if bin_idx in filled_bins:
            continue
        x = bin_idx * 30.0
        rail.append(
            ExpandedObject(
                object_id=rail_ids[i % len(rail_ids)],
                x=float(x),
                y=rail_y[i % len(rail_y)],
                role="structural",
                section_id="section-0",
            )
        )
        filled_bins.add(bin_idx)
        i += 1
    merged = rail + objects
    merged.sort(key=lambda o: (o.x, o.y, o.object_id))
    return merged, len(rail)


def _ensure_trigger_floor(
    objects: list[ExpandedObject],
    *,
    sections: int,
) -> tuple[list[ExpandedObject], int]:
    trigger_count = sum(1 for o in objects if _role_for_object_id(o.object_id) == "trigger")
    expected = max(3, sections)
    missing = max(0, expected - trigger_count)
    if missing == 0 or not objects:
        return objects, 0

    x_min = min(o.x for o in objects)
    x_max = max(o.x for o in objects)
    span = max(90.0, x_max - x_min)
    additions: list[ExpandedObject] = []
    for i in range(missing):
        additions.append(
            ExpandedObject(
                object_id="899",
                x=x_min + span * (i + 1) / (missing + 1),
                y=15.0,
                role="trigger",
                section_id=f"section-{min(i, max(0, sections - 1))}",
            )
        )
    merged = objects + additions
    merged.sort(key=lambda o: (o.x, o.y, o.object_id))
    return merged, len(additions)


def _sample_one(
    model: GMDLanguageModel,
    vocab: IdVocab,
    *,
    sections: int = 4,
    cfg: SamplingConfig,
    seed: int,
) -> _SampleCandidate:
    g = torch.Generator().manual_seed(seed)
    inv_id = _slot_to_object_id(vocab)
    if not inv_id:
        raise RuntimeError("IdVocab is empty; train a model first.")

    device = next(model.parameters()).device
    ctx = model.cfg.ctx
    id_vocab = model.cfg.id_vocab

    # forbid special tokens (PAD/BOS/EOS/UNK) from being sampled as objects.
    # EOS we still allow at every step but only stop after we've produced
    # at least 4 objects per requested section so we do not collapse early.
    forbidden_id = torch.zeros(id_vocab, dtype=torch.bool, device=device)
    forbidden_id[PAD_ID] = True
    forbidden_id[BOS_ID] = True
    forbidden_id[UNK_ID] = True

    initial_mode = _initial_mode_from_prompt(cfg.prompt)

    ids = [BOS_ID]
    cls = [0]
    dx = [0]
    y = [4]
    mode = [initial_mode]
    speed = [DEFAULT_SPEED_IDX]
    section = [0]

    objects: list[ExpandedObject] = []
    cur_x = 0.0
    cur_section = 0

    portal_object_id = mode_portal_object_id(initial_mode)
    portal_slot = vocab.slot(portal_object_id) if portal_object_id else UNK_ID
    if portal_object_id and portal_slot >= NUM_SPECIAL:
        objects.append(
            ExpandedObject(
                object_id=portal_object_id,
                x=0.0,
                y=105.0,
                role="portal",
                section_id="section-0",
            )
        )
        _append_history(
            object_id=portal_object_id,
            id_sample=portal_slot,
            step_dx=0.0,
            cur_y=105.0,
            cur_section=0,
            ids=ids,
            cls=cls,
            dx=dx,
            y=y,
            mode=mode,
            speed=speed,
            section=section,
        )

    target_objects = max(80, sections * 100)
    target_objects = min(target_objects, cfg.max_objects)
    objects_per_section = max(1, target_objects // max(1, sections))

    model.eval()
    with torch.no_grad():
        for _ in range(target_objects):
            window_ids = ids[-ctx:]
            window_cls = cls[-ctx:]
            window_dx = dx[-ctx:]
            window_y = y[-ctx:]
            window_mode = mode[-ctx:]
            window_speed = speed[-ctx:]
            window_section = section[-ctx:]

            batch = {
                "ids": torch.tensor([window_ids], dtype=torch.long, device=device),
                "cls": torch.tensor([window_cls], dtype=torch.long, device=device),
                "dx": torch.tensor([window_dx], dtype=torch.long, device=device),
                "y": torch.tensor([window_y], dtype=torch.long, device=device),
                "mode": torch.tensor([window_mode], dtype=torch.long, device=device),
                "speed": torch.tensor([window_speed], dtype=torch.long, device=device),
                "section": torch.tensor([window_section], dtype=torch.long, device=device),
            }
            out = model(batch)
            last_id = out["logits_id"][0, -1]
            last_dx = out["logits_dx"][0, -1]
            last_y = out["logits_y"][0, -1]
            last_id = _bias_id_logits(
                last_id,
                inv_id=inv_id,
                objects=objects,
                cfg=cfg,
            )
            last_dx = _bias_small_vocab_logits(last_dx, dx)
            last_y = _bias_small_vocab_logits(last_y, y)

            id_logits = _filter_logits(
                last_id, temperature=cfg.temperature,
                top_k=cfg.top_k, top_p=cfg.top_p, forbidden=forbidden_id,
            )
            dx_logits = _filter_logits(
                last_dx, temperature=cfg.temperature,
                top_k=min(cfg.top_k, last_dx.size(-1)), top_p=cfg.top_p,
            )
            y_logits = _filter_logits(
                last_y, temperature=cfg.temperature,
                top_k=min(cfg.top_k, last_y.size(-1)), top_p=cfg.top_p,
            )

            id_probs = F.softmax(id_logits, dim=-1)
            dx_probs = F.softmax(dx_logits, dim=-1)
            y_probs = F.softmax(y_logits, dim=-1)

            id_sample = int(torch.multinomial(id_probs, 1, generator=g).item())
            dx_sample = int(torch.multinomial(dx_probs, 1, generator=g).item())
            y_sample = int(torch.multinomial(y_probs, 1, generator=g).item())

            if id_sample == EOS_ID:
                if len(objects) >= max(40, target_objects // 2):
                    break
                continue
            if id_sample < NUM_SPECIAL:
                continue

            object_id = inv_id.get(id_sample)
            if not object_id:
                continue

            step_dx = max(0.0, _bin_center_dx(dx_sample))
            cur_x += max(cfg.min_dx, step_dx)
            cur_y = float(_bin_center_y(y_sample))
            cur_y = max(0.0, min(540.0, cur_y))
            role = _role_for_object_id(object_id)
            if role == "structural":
                cur_y = _terrain_y(len(objects), cur_section)

            sec_key = f"section-{cur_section}"
            objects.append(
                ExpandedObject(
                    object_id=object_id,
                    x=float(cur_x),
                    y=float(cur_y),
                    role=role,
                    section_id=sec_key,
                )
            )
            _append_history(
                object_id=object_id,
                id_sample=id_sample,
                step_dx=step_dx,
                cur_y=cur_y,
                cur_section=cur_section,
                ids=ids,
                cls=cls,
                dx=dx,
                y=y,
                mode=mode,
                speed=speed,
                section=section,
            )

            if len(objects) % objects_per_section == 0 and cur_section + 1 < sections:
                cur_section += 1

    if not objects:
        for i in range(40):
            objects.append(
                ExpandedObject(
                    object_id=str((i % 6) + 1),
                    x=float(i * 30),
                    y=(90.0, 105.0, 120.0, 105.0)[i % 4],
                    role="structural",
                    section_id="section-0",
                )
            )

    score, diagnostics = _candidate_score(objects, sections=sections)
    return _SampleCandidate(objects=objects, score=score, diagnostics=diagnostics)


def generate_with_diagnostics(
    model: GMDLanguageModel,
    vocab: IdVocab,
    *,
    sections: int = 4,
    cfg: SamplingConfig | None = None,
) -> tuple[list[ExpandedObject], dict[str, Any]]:
    """Sample candidates, select the best structural one, then apply repairs."""
    cfg = cfg or SamplingConfig()
    width = max(1, cfg.candidates)
    candidates = [
        _sample_one(
            model,
            vocab,
            sections=sections,
            cfg=cfg,
            seed=cfg.seed + i * 104_729,
        )
        for i in range(width)
    ]
    best_idx, best = max(enumerate(candidates), key=lambda item: item[1].score)
    objects, rail_added = _add_support_rail(best.objects, cfg=cfg)
    objects, triggers_added = _ensure_trigger_floor(objects, sections=sections)
    diagnostics = {
        **best.diagnostics,
        "candidate_width": width,
        "selected_candidate": best_idx,
        "candidate_scores": [round(c.score, 6) for c in candidates],
        "support_rail_objects_added": rail_added,
        "trigger_floor_objects_added": triggers_added,
        "final_object_count": len(objects),
        "repair_added_ratio": round(
            (rail_added + triggers_added) / max(1, len(objects)),
            6,
        ),
    }
    return objects, diagnostics


def generate(
    model: GMDLanguageModel,
    vocab: IdVocab,
    *,
    sections: int = 4,
    cfg: SamplingConfig | None = None,
) -> list[ExpandedObject]:
    """Sample a level token-by-token and materialize ExpandedObject instances."""
    objects, _diagnostics = generate_with_diagnostics(
        model,
        vocab,
        sections=sections,
        cfg=cfg,
    )
    return objects


def generate_from_checkpoint(
    ckpt_path: Path,
    *,
    sections: int = 4,
    seed: int = 0,
    max_objects: int = 600,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 40,
    prompt: str = "",
    candidates: int = 3,
) -> tuple[list[ExpandedObject], dict[str, Any]]:
    from gmdgen.ml.train import load_checkpoint  # avoid import cycle

    model, vocab, ckpt = load_checkpoint(ckpt_path)
    device = get_best_device()
    model.to(device)
    cfg = SamplingConfig(
        max_objects=max_objects,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        prompt=prompt,
        candidates=candidates,
    )
    objects, diagnostics = generate_with_diagnostics(model, vocab, sections=sections, cfg=cfg)
    device_info = get_device_info()
    info = {
        "n_objects": len(objects),
        "n_params": ckpt.get("n_params", 0),
        "completed_steps": ckpt.get("completed_steps", 0),
        "final_eval": ckpt.get("final_eval", {}),
        "sections": sections,
        "seed": seed,
        "sampling": diagnostics,
        "compute_device": device_info.compute_device,
        "gpu_used_for_generation": device_info.gpu_available,
    }
    return objects, info


__all__ = [
    "SamplingConfig",
    "generate",
    "generate_from_checkpoint",
    "generate_with_diagnostics",
]
