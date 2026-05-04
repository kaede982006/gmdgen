# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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
    """Deep Learning textbook-aligned model plan without adding torch as a hard dep.

    Goodfellow/Bengio/Courville framing:
    - representation learning: separate audio and GD logic factors;
    - CNN/temporal models: local rhythm and spectral patterns;
    - sequence modeling: section/beat/event dependencies;
    - structured output: SectionPlan/SpeedPlan/ObjectPlan/TriggerPlan;
    - regularization: safe simplification and editor repair;
    - optimization: candidate scoring when gradients are unavailable.
    """

    audio_encoder = EncoderSpec(
        name="CNNTransformerAudioEncoder",
        inputs=[
            "mel_spectrogram",
            "onset_curve",
            "beat_grid",
            "energy_envelope",
            "chroma",
            "section_labels",
        ],
        layers=[
            "2d_cnn_over_mel",
            "temporal_cnn_on_onset_energy",
            "beat_position_embedding",
            "transformer_encoder",
        ],
        outputs=[
            "beat_embedding",
            "onset_embedding",
            "section_embedding",
            "energy_embedding",
            "motif_embedding",
        ],
        role="Extract local rhythm, section changes, drops, buildups, and repeated motifs.",
    )
    level_logic_encoder = EncoderSpec(
        name="GDLogicEncoder",
        inputs=[
            "object_tokens",
            "object_id",
            "x_y",
            "group_id",
            "trigger_target",
            "speed_portal",
            "game_mode",
            "editor_layer",
            "color_channel",
            "object_role",
            "section_id",
        ],
        layers=[
            "object_embedding",
            "relative_x_time_embedding",
            "group_trigger_graph_encoder",
            "section_transformer_encoder",
        ],
        outputs=[
            "style_embedding",
            "motif_embedding",
            "trigger_graph_embedding",
            "speed_context_embedding",
        ],
        role="Learn reference-level style, GD object structure, trigger/group relations, and speed context.",
    )
    fusion = FusionSpec(
        name="AudioGDConditioningFusion",
        inputs=[
            "beat_embedding",
            "section_embedding",
            "energy_embedding",
            "onset_embedding",
            "style_embedding",
            "speed_context_embedding",
        ],
        operations=[
            "cross_attention_audio_to_level",
            "beat_event_candidate_scoring",
            "section_density_conditioning",
            "speed_state_conditioning",
            "style_motif_selection",
        ],
        outputs=[
            "section_plan_latents",
            "speed_plan_latents",
            "gameplay_event_latents",
            "trigger_event_latents",
        ],
    )
    generator = StructuredGeneratorSpec(
        outputs=[
            "SectionPlan",
            "SpeedPlan",
            "GameplayEventPlan",
            "ObjectPlan",
            "TriggerPlan",
            "ValidationReport",
        ],
        decoding_strategy="beam_search_or_candidate_search_with_editor_repair",
        constraints=[
            "posForTime/timeForPos consistency",
            "speed object order",
            "trigger target existence",
            "group id bounds",
            "object budget",
            "editor-safe save-string encoding",
            "playability spacing",
        ],
    )
    losses = [
        LossTermSpec("L_beat_sync", "Gameplay events near beats.", "nearest beat error", 18.0),
        LossTermSpec("L_onset_sync", "Visual triggers near onsets.", "nearest onset error", 10.0),
        LossTermSpec("L_section_sync", "GD section changes near music section boundaries.", "boundary match", 8.0),
        LossTermSpec("L_time_to_x_consistency", "X and audio time invert correctly.", "pos/time roundtrip", 18.0, True),
        LossTermSpec("L_speed_portal_consistency", "Speed portals force beat-to-X remapping.", "speed segment roundtrip", 12.0, True),
        LossTermSpec("L_energy_density", "Object density follows energy envelope.", "density-energy correlation", 8.0),
        LossTermSpec("L_style_consistency", "Match reference object/style distribution.", "class/style divergence", 6.0),
        LossTermSpec("L_playability", "Avoid impossible spacing and unsafe transitions.", "rule simulation penalties", 12.0, True),
        LossTermSpec("L_trigger_validity", "Trigger targets and durations are valid.", "trigger graph checks", 14.0, True),
        LossTermSpec("L_group_validity", "Group IDs exist and stay in bounds.", "group graph checks", 12.0, True),
        LossTermSpec("L_editor_validity", "Editor can load, preview, and save.", "roundtrip/import checks", 10.0, True),
        LossTermSpec("L_object_budget", "Stay under object/capacity budget.", "count budget", 8.0, True),
    ]
    return AudioConditionedModelSpec(
        audio_encoder=audio_encoder,
        level_logic_encoder=level_logic_encoder,
        fusion=fusion,
        structured_generator=generator,
        losses=losses,
    )
