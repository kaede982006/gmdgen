from __future__ import annotations

from gmdgen.representation.object_classifier import (
    ObjectClass,
    class_short,
    classify,
    is_structural,
    is_visible,
)


def test_speed_portal_classified_as_portal() -> None:
    assert classify("200") == ObjectClass.PORTAL
    assert classify("201") == ObjectClass.PORTAL
    assert classify("1334") == ObjectClass.PORTAL


def test_gamemode_portal_classified_as_portal() -> None:
    assert classify("12") == ObjectClass.PORTAL   # ship
    assert classify("47") == ObjectClass.PORTAL   # ufo


def test_trigger_classified_as_trigger() -> None:
    assert classify("29") == ObjectClass.TRIGGER  # color trigger
    assert classify("30") == ObjectClass.TRIGGER  # move trigger
    assert classify("899") == ObjectClass.TRIGGER # stop trigger


def test_basic_block_classified_as_structure() -> None:
    assert classify("1") == ObjectClass.STRUCTURE
    assert classify("4") == ObjectClass.STRUCTURE


def test_mid_range_id_classified_as_decoration() -> None:
    assert classify("500") == ObjectClass.DECORATION
    assert classify("900") == ObjectClass.DECORATION


def test_unknown_for_zero_and_string() -> None:
    assert classify("0") == ObjectClass.UNKNOWN
    assert classify("abc") == ObjectClass.UNKNOWN


def test_is_structural() -> None:
    assert is_structural("1")    # structure
    assert is_structural("200")  # portal
    assert not is_structural("500")  # decoration
    assert not is_structural("29")   # trigger


def test_is_visible() -> None:
    assert is_visible("1")
    assert is_visible("500")
    assert not is_visible("29")   # trigger not visible


def test_class_short_returns_single_char() -> None:
    for id_str in ["1", "29", "200", "12", "500"]:
        short = class_short(id_str)
        assert len(short) == 1
        assert short in "SDTPXU"


def test_class_short_matches_classify() -> None:
    mapping = {
        ObjectClass.STRUCTURE: "S",
        ObjectClass.DECORATION: "D",
        ObjectClass.TRIGGER: "T",
        ObjectClass.PORTAL: "P",
        ObjectClass.SPECIAL: "X",
        ObjectClass.UNKNOWN: "U",
    }
    for id_str in ["1", "500", "29", "200", "0", "35"]:
        cls = classify(id_str)
        assert class_short(id_str) == mapping[cls]
