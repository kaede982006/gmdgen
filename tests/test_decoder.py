from __future__ import annotations

from gmdgen.io.gmd_decoder import (
    decode_level_data,
    decode_level_description,
    encode_level_data,
    encode_level_description,
)


def test_decode_encode_roundtrip() -> None:
    level_data = "1,1,2,30,3,90;1,2,2,60,3,120;"
    encoded = encode_level_data(level_data)
    decoded = decode_level_data(encoded)
    assert decoded == level_data
    assert len(encoded) % 4 == 0


def test_description_encode_decode_roundtrip() -> None:
    description = "Prompt: wave hell memory"
    encoded = encode_level_description(description)
    decoded = decode_level_description(encoded)
    assert decoded == description
