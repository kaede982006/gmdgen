from __future__ import annotations

from gmdgen.data.preprocess import (
    detect_section_boundaries,
    objects_cross_portal,
)
from gmdgen.representation.tokenizer import (
    level_data_to_feature_tokens,
    to_feature_token,
)


def _obj(object_id: str, x: int, y: int) -> str:
    return f"1,{object_id},2,{x},3,{y}"


def _portal(object_id: str, x: int) -> str:
    return f"1,{object_id},2,{x},3,180"


# ── Section boundary detection ────────────────────────────────────────────────

def test_no_portals_single_boundary() -> None:
    objects = [_obj("1", 30, 180), _obj("1", 60, 180)]
    boundaries = detect_section_boundaries(objects)
    assert len(boundaries) == 1
    assert boundaries[0].start_object_index == 0
    assert boundaries[0].gamemode == "cube"
    assert boundaries[0].speed == "normal"


def test_gamemode_portal_adds_boundary() -> None:
    objects = [
        _obj("1", 30, 180),
        _portal("12", 60),   # ship portal
        _obj("1", 90, 180),
    ]
    boundaries = detect_section_boundaries(objects)
    assert len(boundaries) == 2
    assert boundaries[1].gamemode == "ship"


def test_speed_portal_adds_boundary() -> None:
    objects = [
        _obj("1", 30, 180),
        _portal("202", 60),  # double speed
        _obj("1", 90, 180),
    ]
    boundaries = detect_section_boundaries(objects)
    assert len(boundaries) == 2
    assert boundaries[1].speed == "double"


def test_multiple_portals_multiple_boundaries() -> None:
    objects = [
        _obj("1", 0, 180),
        _portal("202", 30),   # speed double
        _obj("1", 60, 180),
        _portal("12", 90),    # ship
        _obj("1", 120, 180),
    ]
    boundaries = detect_section_boundaries(objects)
    assert len(boundaries) == 3


# ── Cross-portal detection ────────────────────────────────────────────────────

def test_chunk_crosses_portal() -> None:
    objects = [_obj("1", 0, 180)] * 3 + [_portal("12", 90)] + [_obj("1", 120, 180)] * 3
    assert objects_cross_portal(objects, start=2, size=4) is True


def test_chunk_does_not_cross_portal() -> None:
    objects = [_obj("1", 0, 180)] * 3 + [_portal("12", 90)] + [_obj("1", 120, 180)] * 3
    assert objects_cross_portal(objects, start=4, size=3) is False


# ── Feature tokeniser ─────────────────────────────────────────────────────────

def test_feature_token_format() -> None:
    obj = _obj("1", 30, 180)
    token = to_feature_token(obj, previous_x=0.0, section_id=0)
    assert token is not None
    assert token.startswith("OBJ:1|")
    assert "|CLS:" in token
    assert "|DX:" in token
    assert "|Y:" in token
    assert "|SEC:0" in token


def test_feature_token_class_structure() -> None:
    obj = _obj("1", 60, 180)
    token = to_feature_token(obj, previous_x=30.0, section_id=0)
    assert token is not None
    assert "|CLS:S|" in token


def test_feature_token_class_trigger() -> None:
    trigger_obj = f"1,29,2,100,3,180,51,5"
    token = to_feature_token(trigger_obj, previous_x=90.0, section_id=1)
    assert token is not None
    assert "|CLS:T|" in token
    assert "|SEC:1" in token


def test_level_data_to_feature_tokens_eos() -> None:
    level_data = f"kA11,0;{_obj('1', 30, 180)};{_obj('2', 60, 180)};"
    tokens = level_data_to_feature_tokens(level_data)
    assert tokens[-1] == "<EOS>"
    assert any(t.startswith("OBJ:") for t in tokens)


def test_feature_tokens_include_portal_section_change() -> None:
    level_data = (
        "kA11,0;"
        + _obj("1", 30, 180) + ";"
        + _portal("202", 60) + ";"   # speed portal
        + _obj("1", 90, 180) + ";"
    )
    tokens = level_data_to_feature_tokens(level_data)
    sec_ids = {t.split("|SEC:")[-1] for t in tokens if "|SEC:" in t}
    assert len(sec_ids) >= 2, "Objects in different sections should have different section IDs"
