# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

"""Tests for Phase 10: GD level description encoding correctness.

The previous failure mode was: in-game description showed raw base64 text
because the description was double-encoded or because no decode happened
on inspection. These tests verify single-encoding round-trip and that
the description used during export is plain human-readable text."""

import base64
import re

from gmdgen.io.gmd_decoder import decode_level_description, encode_level_description


def test_description_single_encode_roundtrip():
    plain = "Generated Geometry Dash level."
    encoded = encode_level_description(plain)
    decoded = decode_level_description(encoded)
    assert decoded == plain, f"Roundtrip failed: {plain!r} -> {encoded!r} -> {decoded!r}"


def test_description_no_double_encoding():
    plain = "Test description"
    once = encode_level_description(plain)
    twice = encode_level_description(once)
    # Decoded twice must give back the once-encoded value (which is base64).
    decoded_once = decode_level_description(twice)
    assert decoded_once == once
    # Decoded with a single decode of the once-encoded value gives plain text.
    assert decode_level_description(once) == plain


def test_description_korean_utf8_safe():
    plain = "한국어 설명입니다."
    encoded = encode_level_description(plain)
    decoded = decode_level_description(encoded)
    assert decoded == plain


def test_description_empty_string():
    encoded = encode_level_description("")
    assert encoded == ""
    decoded = decode_level_description(encoded)
    assert decoded == ""


def test_decode_passthrough_for_non_base64():
    """When given non-base64 text, decoder should not garble it."""
    plain = "not base64 here!!"
    decoded = decode_level_description(plain)
    # Either returns input unchanged (preferred) or doesn't produce mojibake.
    assert "�" not in decoded


def test_generator_description_is_plain_human_readable():
    """The deterministic generator description must be plain text."""
    from gmdgen.generate.generator import _build_generated_description
    desc = _build_generated_description("any prompt", ["ref1", "ref2"])
    assert isinstance(desc, str) and len(desc) > 0
    # Must not contain JSON, prompt leak, or base64 noise.
    assert "{" not in desc and "}" not in desc
    # Must not look like base64 (long alphanumeric run with no spaces).
    assert " " in desc or len(desc) < 30, f"Description looks base64-like: {desc!r}"


def test_description_in_export_is_base64_only_once():
    """The k3 tag value should be base64; decoding once gives plain text."""
    plain = "Audio-conditioned GD level."
    encoded_once = encode_level_description(plain)
    # The encoded value should be valid base64
    try:
        decoded_bytes = base64.b64decode(encoded_once + ("=" * (-len(encoded_once) % 4)))
        decoded_text = decoded_bytes.decode("utf-8")
    except Exception as exc:
        raise AssertionError(f"k3 tag is not valid base64: {exc}")
    assert decoded_text == plain
    # Verify the encoded value is not itself plain ASCII text mistakenly
    assert encoded_once != plain


def test_description_no_mojibake_after_export_roundtrip():
    """Special chars survive a full export/import cycle without mojibake."""
    test_strings = [
        "Generated Geometry Dash level.",
        "Тест",
        "テスト",
        "Test with — em-dash and ‘quotes’",
    ]
    for plain in test_strings:
        encoded = encode_level_description(plain)
        decoded = decode_level_description(encoded)
        assert decoded == plain, f"Roundtrip failed for: {plain!r}"
        assert "�" not in decoded


def test_description_does_not_contain_raw_json():
    """Description must not be a raw JSON dump."""
    from gmdgen.generate.generator import _build_generated_description
    desc = _build_generated_description('{"prompt": "test"}', [])
    # The deterministic implementation returns a fixed string; verify it's not JSON.
    assert not (desc.startswith("{") and desc.endswith("}"))
    assert not desc.startswith("[")


def test_long_base64_string_pattern_is_not_in_description():
    """Description must not look like a long base64 string (common leak symptom)."""
    from gmdgen.generate.generator import _build_generated_description
    desc = _build_generated_description("test prompt", [])
    # A "base64-like" string is ≥40 chars of pure base64 alphabet.
    base64_run = re.search(r"[A-Za-z0-9+/=]{40,}", desc)
    assert base64_run is None, f"Description contains suspected base64 leak: {desc!r}"
