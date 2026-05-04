from __future__ import annotations

import base64
import gzip
import zlib


def _add_padding(value: str) -> str:
    return value + ("=" * (-len(value) % 4))


def decode_level_data(encoded_value: str) -> str:
    encoded_value = encoded_value.strip()
    if not encoded_value:
        return ""

    payload = base64.urlsafe_b64decode(_add_padding(encoded_value))

    try:
        decoded_bytes = gzip.decompress(payload)
    except OSError:
        decoded_bytes = zlib.decompress(payload, zlib.MAX_WBITS | 32)

    return decoded_bytes.decode("utf-8", errors="replace")


def encode_level_data(decoded_value: str) -> str:
    compressed = gzip.compress(decoded_value.encode("utf-8"))
    # Keep base64 padding because GMD importers may not auto-repair missing "=".
    return base64.urlsafe_b64encode(compressed).decode("ascii")


def decode_level_description(encoded_value: str) -> str:
    encoded_value = encoded_value.strip()
    if not encoded_value:
        return ""

    try:
        decoded = base64.b64decode(_add_padding(encoded_value))
    except Exception:  # noqa: BLE001
        return encoded_value

    text = decoded.decode("utf-8", errors="replace")
    control_chars = sum(1 for ch in text if ord(ch) < 9 or (13 < ord(ch) < 32))
    if control_chars > 3:
        return encoded_value
    return text


def encode_level_description(description: str) -> str:
    return base64.b64encode(description.encode("utf-8")).decode("ascii")
