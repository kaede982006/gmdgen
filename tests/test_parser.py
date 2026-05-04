# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from gmdgen.io.gmd_parser import parse_gmd_text
from gmdgen.io.gmd_writer import serialize_tags


def test_parse_and_serialize_roundtrip() -> None:
    raw = "<d><k>k2</k><s>Sample</s><k>k95</k><i>3</i></d>"
    parsed = parse_gmd_text(raw)

    assert parsed["k2"] == ("s", "Sample")
    assert parsed["k95"] == ("i", "3")

    rebuilt = serialize_tags(parsed)
    reparsed = parse_gmd_text(rebuilt)
    assert reparsed == parsed


def test_parse_plist_with_bool_nodes() -> None:
    raw = (
        '<?xml version="1.0"?>'
        '<plist version="1.0" gjver="2.0"><dict>'
        "<k>k2</k><s>Sample</s>"
        "<k>k65</k><t />"
        "<k>k95</k><i>3</i>"
        "</dict></plist>"
    )
    parsed = parse_gmd_text(raw)

    assert parsed["k2"] == ("s", "Sample")
    assert parsed["k65"] == ("t", "")
    assert parsed["k95"] == ("i", "3")
