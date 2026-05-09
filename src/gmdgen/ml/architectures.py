# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 kaede982006
"""GMD language model architectures.

This module ships *both* a textual model spec (kept for backward compatibility
with the existing report pipeline) and a *real* PyTorch ``nn.Module``
implementation of a small causal Transformer over factorized GMD tokens.

The Transformer follows Fleuret §5.3 / §5.7:
  * factorized embedding (Fleuret §4.9): id + cls + dx + y + mode + speed +
    section embeddings are summed into a d_model vector.
  * sinusoidal positional encoding (Fleuret §4.10).
  * stacked TransformerEncoderLayer blocks with a causal mask, equivalent to a
    GPT-style decoder when bidirectional self-attention is masked.
  * three independent linear heads predict next id, next dx-bin, next y-bin.

The model is intentionally *small* (~1–3M parameters) so it can be trained on
a CPU within minutes from the 129-file dataset.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

try:  # torch is an optional dependency; the spec helpers must keep working.
    import torch
    import torch.nn as nn
    from torch import Tensor

    _TORCH_OK = True
except Exception:  # pragma: no cover - exercised only when torch is missing
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[assignment, misc]
    _TORCH_OK = False

from gmdgen.ml.tokens import (
    CLS_VOCAB,
    DX_VOCAB,
    ID_VOCAB_SIZE,
    MODE_VOCAB,
    PAD_ID,
    SECTION_VOCAB,
    SPEED_VOCAB,
    Y_VOCAB,
)


# ─────────────────────────────────────────────────────────────────────────────
# Legacy spec dataclasses (kept for backward compatibility with reports/tests)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class EncoderSpec:
    name: str
    inputs: list[str]
    layers: list[str]
    outputs: list[str]
    role: str


@dataclass(slots=True)
class FusionSpec:
    name: str
    inputs: list[str]
    operations: list[str]
    outputs: list[str]


@dataclass(slots=True)
class StructuredGeneratorSpec:
    outputs: list[str]
    decoding_strategy: str
    constraints: list[str]


@dataclass(slots=True)
class LossTermSpec:
    name: str
    purpose: str
    signal: str
    default_weight: float
    is_hard_constraint: bool = False


@dataclass(slots=True)
class AudioConditionedModelSpec:
    audio_encoder: EncoderSpec
    level_logic_encoder: EncoderSpec
    fusion: FusionSpec
    structured_generator: StructuredGeneratorSpec
    losses: list[LossTermSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_encoder": asdict(self.audio_encoder),
            "level_logic_encoder": asdict(self.level_logic_encoder),
            "fusion": asdict(self.fusion),
            "structured_generator": asdict(self.structured_generator),
            "losses": [asdict(loss) for loss in self.losses],
        }


def build_audio_conditioned_model_spec() -> AudioConditionedModelSpec:
    """Deep Learning textbook-aligned model plan, used as a release report.

    The actual trained model is :class:`GMDLanguageModel`; this spec only
    documents the longer-term audio-conditioned design.
    """
    audio_encoder = EncoderSpec(
        name="CNNTransformerAudioEncoder",
        inputs=["mel_spectrogram", "onset_curve", "beat_grid", "energy_envelope"],
        layers=["2d_cnn_over_mel", "temporal_cnn_on_onset_energy",
                "beat_position_embedding", "transformer_encoder"],
        outputs=["beat_embedding", "onset_embedding", "section_embedding",
                 "energy_embedding", "motif_embedding"],
        role="Extract local rhythm, section changes, drops, buildups, motifs.",
    )
    level_logic_encoder = EncoderSpec(
        name="GDLogicEncoder",
        inputs=["object_tokens", "object_id", "x_y", "group_id",
                "trigger_target", "speed_portal", "game_mode", "section_id"],
        layers=["object_embedding", "relative_x_time_embedding",
                "group_trigger_graph_encoder", "section_transformer_encoder"],
        outputs=["style_embedding", "motif_embedding",
                 "trigger_graph_embedding", "speed_context_embedding"],
        role="Learn level style, GD object structure, trigger/group relations.",
    )
    fusion = FusionSpec(
        name="AudioGDConditioningFusion",
        inputs=["beat_embedding", "section_embedding", "energy_embedding",
                "onset_embedding", "style_embedding", "speed_context_embedding"],
        operations=["cross_attention_audio_to_level",
                    "beat_event_candidate_scoring",
                    "section_density_conditioning",
                    "speed_state_conditioning",
                    "style_motif_selection"],
        outputs=["section_plan_latents", "speed_plan_latents",
                 "gameplay_event_latents", "trigger_event_latents"],
    )
    generator = StructuredGeneratorSpec(
        outputs=["SectionPlan", "SpeedPlan", "GameplayEventPlan",
                 "ObjectPlan", "TriggerPlan", "ValidationReport"],
        decoding_strategy="beam_search_or_candidate_search_with_editor_repair",
        constraints=["posForTime/timeForPos consistency", "speed object order",
                     "trigger target existence", "group id bounds",
                     "object budget", "editor-safe save-string encoding",
                     "playability spacing"],
    )
    losses = [
        LossTermSpec("L_beat_sync", "Gameplay events near beats.", "nearest beat error", 18.0),
        LossTermSpec("L_onset_sync", "Visual triggers near onsets.", "nearest onset error", 10.0),
        LossTermSpec("L_section_sync", "GD section changes near music boundaries.", "boundary match", 8.0),
        LossTermSpec("L_time_to_x_consistency", "X and audio time invert correctly.", "pos/time roundtrip", 18.0, True),
        LossTermSpec("L_speed_portal_consistency", "Speed portals force beat-to-X remapping.", "speed segment roundtrip", 12.0, True),
        LossTermSpec("L_energy_density", "Object density follows energy envelope.", "density-energy correlation", 8.0),
        LossTermSpec("L_style_consistency", "Match reference object/style distribution.", "class/style divergence", 6.0),
        LossTermSpec("L_playability", "Avoid impossible spacing.", "rule simulation penalties", 12.0, True),
        LossTermSpec("L_trigger_validity", "Trigger targets/durations valid.", "trigger graph checks", 14.0, True),
        LossTermSpec("L_group_validity", "Group IDs exist and stay in bounds.", "group graph checks", 12.0, True),
        LossTermSpec("L_editor_validity", "Editor can load, preview, save.", "roundtrip/import checks", 10.0, True),
        LossTermSpec("L_object_budget", "Stay under capacity.", "count budget", 8.0, True),
    ]
    return AudioConditionedModelSpec(
        audio_encoder=audio_encoder,
        level_logic_encoder=level_logic_encoder,
        fusion=fusion,
        structured_generator=generator,
        losses=losses,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Real model: GMDLanguageModel — small causal Transformer (Fleuret §5.3/§5.7)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ModelConfig:
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    d_ff: int = 256
    ctx: int = 512
    dropout: float = 0.1

    id_vocab: int = ID_VOCAB_SIZE
    cls_vocab: int = CLS_VOCAB
    dx_vocab: int = DX_VOCAB
    y_vocab: int = Y_VOCAB
    mode_vocab: int = MODE_VOCAB
    speed_vocab: int = SPEED_VOCAB
    section_vocab: int = SECTION_VOCAB

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_torch() -> None:
    if not _TORCH_OK:
        raise RuntimeError(
            "PyTorch is required for the GMDLanguageModel. Install with "
            "`pip install gmdgen[ml]` or `pip install torch numpy`."
        )


if _TORCH_OK:

    class _SinusoidalPositionalEncoding(nn.Module):
        """Fleuret §4.10 — fixed sinusoidal positional encoding."""

        def __init__(self, d_model: int, max_len: int = 4096) -> None:
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float)
                * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))  # [1, L, D]

        def forward(self, x: Tensor) -> Tensor:
            # x: [B, T, D]
            t = x.size(1)
            return x + self.pe[:, :t]

    class GMDLanguageModel(nn.Module):
        """Causal Transformer LM over factorized GMD tokens.

        Inputs are 7 parallel index streams (id/cls/dx/y/mode/speed/section).
        The factorized embeddings are summed (Fleuret §4.9 — disentangled
        factor embedding: lets the model share strength across rare combos).
        Three linear heads predict next id, next dx-bin, next y-bin.
        """

        def __init__(self, cfg: ModelConfig) -> None:
            super().__init__()
            self.cfg = cfg

            # one embedding table per factor
            self.emb_id = nn.Embedding(cfg.id_vocab, cfg.d_model, padding_idx=PAD_ID)
            self.emb_cls = nn.Embedding(cfg.cls_vocab, cfg.d_model)
            self.emb_dx = nn.Embedding(cfg.dx_vocab, cfg.d_model)
            self.emb_y = nn.Embedding(cfg.y_vocab, cfg.d_model)
            self.emb_mode = nn.Embedding(cfg.mode_vocab, cfg.d_model)
            self.emb_speed = nn.Embedding(cfg.speed_vocab, cfg.d_model)
            self.emb_section = nn.Embedding(cfg.section_vocab, cfg.d_model)

            self.pos = _SinusoidalPositionalEncoding(cfg.d_model, max_len=cfg.ctx + 8)
            self.dropout = nn.Dropout(cfg.dropout)

            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_head,
                dim_feedforward=cfg.d_ff,
                dropout=cfg.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.n_layer)
            self.norm = nn.LayerNorm(cfg.d_model)

            self.head_id = nn.Linear(cfg.d_model, cfg.id_vocab)
            self.head_dx = nn.Linear(cfg.d_model, cfg.dx_vocab)
            self.head_y = nn.Linear(cfg.d_model, cfg.y_vocab)

            # weight init similar to GPT-2
            self.apply(self._init_weights)

        @staticmethod
        def _init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        def num_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        def _embed(
            self,
            ids: Tensor,
            cls: Tensor,
            dx: Tensor,
            y: Tensor,
            mode: Tensor,
            speed: Tensor,
            section: Tensor,
        ) -> Tensor:
            x = (
                self.emb_id(ids)
                + self.emb_cls(cls)
                + self.emb_dx(dx)
                + self.emb_y(y)
                + self.emb_mode(mode)
                + self.emb_speed(speed)
                + self.emb_section(section)
            )
            return self.dropout(self.pos(x))

        def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
            ids = batch["ids"]
            T = ids.size(1)
            x = self._embed(
                ids,
                batch["cls"],
                batch["dx"],
                batch["y"],
                batch["mode"],
                batch["speed"],
                batch["section"],
            )
            mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            key_padding_mask = ids.eq(PAD_ID)
            h = self.blocks(x, mask=mask, src_key_padding_mask=key_padding_mask)
            h = self.norm(h)
            return {
                "logits_id": self.head_id(h),
                "logits_dx": self.head_dx(h),
                "logits_y": self.head_y(h),
                "h": h,
            }


else:  # pragma: no cover

    class GMDLanguageModel:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()


def build_default_model() -> "GMDLanguageModel":
    _require_torch()
    return GMDLanguageModel(ModelConfig())


__all__ = [
    "EncoderSpec",
    "FusionSpec",
    "StructuredGeneratorSpec",
    "LossTermSpec",
    "AudioConditionedModelSpec",
    "build_audio_conditioned_model_spec",
    "ModelConfig",
    "GMDLanguageModel",
    "build_default_model",
]
