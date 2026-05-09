# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 kaede982006
"""GMD token dataset for language model training.

We do *self-supervised next-object prediction* (Foundations of LLMs §1.1.1).
A long .gmd file is sliced into overlapping fixed-length windows; each window
is the input, and the same window shifted by 1 is the target.

Augmentation (Fleuret §3.7 — benefits of scale; we do not have scale, so we
multiply the dataset effectively):

  * y_mirror      : negate y around the gameplay corridor center (~180).
  * section_shuffle: reorder whole sections (their internal x re-laid).
  * dropout_decor : drop a random subset of decoration tokens.

These are applied with low probability so the underlying distribution is
preserved.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from gmdgen.data.loader import load_dataset
from gmdgen.ml.tokens import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    FactorizedToken,
    IdVocab,
    build_id_vocab,
    encode_level_string,
    tokens_to_id_array,
)


@dataclass(slots=True)
class DatasetConfig:
    ctx: int = 512
    stride: int = 256
    augment: bool = True
    p_y_mirror: float = 0.15
    p_section_shuffle: float = 0.10
    p_drop_decor: float = 0.10
    seed: int = 0


def _y_mirror(tokens: list[FactorizedToken]) -> list[FactorizedToken]:
    """Mirror y around the gameplay-corridor center to inflate the dataset."""
    out = []
    for t in tokens:
        # bin index; mirror by inverting around the central bin (4)
        new_y = max(0, min(8, 8 - t.y_idx))
        out.append(
            FactorizedToken(
                id_idx=t.id_idx,
                cls_idx=t.cls_idx,
                dx_idx=t.dx_idx,
                y_idx=new_y,
                mode_idx=t.mode_idx,
                speed_idx=t.speed_idx,
                section_idx=t.section_idx,
                raw_object_id=t.raw_object_id,
                raw_x=t.raw_x,
                raw_y=-t.raw_y + 360.0,
            )
        )
    return out


def _drop_decoration(
    tokens: list[FactorizedToken], rng: random.Random, p: float
) -> list[FactorizedToken]:
    out = []
    for t in tokens:
        if t.cls_idx == 2 and rng.random() < p:  # decoration
            continue
        out.append(t)
    return out


def _section_shuffle(
    tokens: list[FactorizedToken], rng: random.Random
) -> list[FactorizedToken]:
    """Shuffle whole-section spans then re-index sections."""
    if not tokens:
        return tokens
    spans: list[list[FactorizedToken]] = []
    cur: list[FactorizedToken] = []
    cur_sec = tokens[0].section_idx
    for t in tokens:
        if t.section_idx != cur_sec:
            spans.append(cur)
            cur = []
            cur_sec = t.section_idx
        cur.append(t)
    if cur:
        spans.append(cur)
    if len(spans) < 2:
        return tokens
    rng.shuffle(spans)
    flat: list[FactorizedToken] = []
    for new_sec, span in enumerate(spans):
        for t in span:
            flat.append(
                FactorizedToken(
                    id_idx=t.id_idx,
                    cls_idx=t.cls_idx,
                    dx_idx=t.dx_idx,
                    y_idx=t.y_idx,
                    mode_idx=t.mode_idx,
                    speed_idx=t.speed_idx,
                    section_idx=min(new_sec, 31),
                    raw_object_id=t.raw_object_id,
                    raw_x=t.raw_x,
                    raw_y=t.raw_y,
                )
            )
    return flat


def _add_bos_eos(tokens: list[FactorizedToken]) -> list[FactorizedToken]:
    bos = FactorizedToken(BOS_ID, 0, 0, 4, 0, 1, 0)
    eos = FactorizedToken(EOS_ID, 0, 0, 4, 0, 1, 0)
    return [bos, *tokens, eos]


def encode_records_to_streams(
    dataset_dir: Path,
    *,
    vocab: IdVocab | None = None,
) -> tuple[list[list[FactorizedToken]], IdVocab]:
    """Decode all .gmd in `dataset_dir` and return one token-stream per file."""
    records = load_dataset(dataset_dir)
    if vocab is None:
        vocab = build_id_vocab(r.decoded_level_data for r in records)
    streams: list[list[FactorizedToken]] = []
    for r in records:
        tokens = encode_level_string(r.decoded_level_data, vocab)
        if tokens:
            streams.append(_add_bos_eos(tokens))
    return streams, vocab


class GMDTokenDataset(Dataset):
    def __init__(
        self,
        streams: list[list[FactorizedToken]],
        cfg: DatasetConfig,
    ) -> None:
        self.cfg = cfg
        self.streams = streams
        self.windows: list[tuple[int, int]] = []
        for si, stream in enumerate(streams):
            n = len(stream)
            if n <= 1:
                continue
            i = 0
            while i < n:
                self.windows.append((si, i))
                if i + cfg.ctx >= n:
                    break
                i += cfg.stride
        # Ensure at least one window even if everything was tiny.
        if not self.windows and streams:
            self.windows.append((0, 0))

    def __len__(self) -> int:
        return len(self.windows)

    def _get_tokens(self, si: int, start: int) -> list[FactorizedToken]:
        stream = self.streams[si]
        rng = random.Random(self.cfg.seed * 65537 + si * 131 + start)
        if self.cfg.augment:
            if rng.random() < self.cfg.p_y_mirror:
                stream = _y_mirror(stream)
            if rng.random() < self.cfg.p_drop_decor:
                stream = _drop_decoration(stream, rng, p=0.3)
            if rng.random() < self.cfg.p_section_shuffle:
                stream = _section_shuffle(stream, rng)
        end = min(start + self.cfg.ctx + 1, len(stream))
        return stream[start:end]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        si, start = self.windows[idx]
        tokens = self._get_tokens(si, start)
        if len(tokens) < 2:
            tokens = tokens + tokens  # pad-by-repetition keeps shapes valid
        arr = tokens_to_id_array(tokens)
        ctx = self.cfg.ctx
        # Build a length-(ctx+1) window so input/target each have length ctx.
        def _pad(seq: list[int]) -> list[int]:
            seq = seq[: ctx + 1]
            return seq + [PAD_ID] * (ctx + 1 - len(seq))

        ids = torch.tensor(_pad(arr["id"]), dtype=torch.long)
        cls = torch.tensor(_pad(arr["cls"]), dtype=torch.long)
        dx = torch.tensor(_pad(arr["dx"]), dtype=torch.long)
        y = torch.tensor(_pad(arr["y"]), dtype=torch.long)
        mode = torch.tensor(_pad(arr["mode"]), dtype=torch.long)
        speed = torch.tensor(_pad(arr["speed"]), dtype=torch.long)
        section = torch.tensor(_pad(arr["section"]), dtype=torch.long)
        return {
            "ids": ids,
            "cls": cls,
            "dx": dx,
            "y": y,
            "mode": mode,
            "speed": speed,
            "section": section,
        }


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


__all__ = [
    "DatasetConfig",
    "GMDTokenDataset",
    "collate",
    "encode_records_to_streams",
]
