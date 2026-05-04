# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from gmdgen.ml.architectures import build_audio_conditioned_model_spec


def test_audio_conditioned_model_spec_contains_structured_outputs() -> None:
    spec = build_audio_conditioned_model_spec()
    payload = spec.to_dict()

    assert "mel_spectrogram" in payload["audio_encoder"]["inputs"]
    assert "trigger_target" in payload["level_logic_encoder"]["inputs"]
    assert "ObjectPlan" in payload["structured_generator"]["outputs"]
    assert "TriggerPlan" in payload["structured_generator"]["outputs"]
    assert any(loss["name"] == "L_time_to_x_consistency" for loss in payload["losses"])
