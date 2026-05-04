from __future__ import annotations

from gmdgen.generate.scoring import compute_audio_conditioned_score
from gmdgen.generate.validator import (
    round_trip_validate,
    validate_encoder_safe_keys,
)
from gmdgen.io.gmd_decoder import encode_level_data


class _Features:
    beat_times = [0.0, 0.5, 1.0]
    onset_times = [0.0, 0.5, 1.0]
    sections = []


def test_round_trip_preserves_object_count() -> None:
    decoded = "kA11,0;1,1,2,30,3,180;1,1,2,60,3,180;"
    report = round_trip_validate(decoded)
    assert report["valid"] is True
    assert report["object_count_before"] == report["object_count_after"]


def test_invalid_key_is_rejected() -> None:
    issues = validate_encoder_safe_keys(["1,1,2,nan,3,180"], safe_mode=True)
    assert any("invalid_numeric_value" in issue for issue in issues)


def test_unsupported_trigger_not_encoded_in_safe_mode() -> None:
    issues = validate_encoder_safe_keys(["1,1347,2,30,3,180,51,1"], safe_mode=True)
    assert any("unsupported_trigger_in_mode" in issue for issue in issues)


def test_nan_coordinate_rejected() -> None:
    report = round_trip_validate("kA11,0;1,1,2,nan,3,180;", safe_mode=True)
    assert report["valid"] is False
    assert any("invalid_numeric_value" in issue for issue in report["issues"])


def test_roundtrip_failure_lowers_editor_validity_score() -> None:
    objects = ["1,1,2,30,3,180"]
    clean = compute_audio_conditioned_score(
        objects,
        audio_features=_Features(),
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=10,
    )
    dirty = compute_audio_conditioned_score(
        objects,
        audio_features=_Features(),
        speed_objects=[],
        start_speed="normal",
        song_offset=0.0,
        beat_snap_tolerance=0.1,
        object_budget=10,
        editor_issues=["round_trip_count_mismatch"],
    )
    assert dirty.editor_validity < clean.editor_validity
