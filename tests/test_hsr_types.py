# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 kaede982006
"""Tests for the LevelPlan / Section / Transitions schema."""
from __future__ import annotations

import json

import pytest

from gmdgen.types import (
    LEVEL_PLAN_JSON_SCHEMA,
    LevelMeta,
    LevelPlan,
    Section,
    Transitions,
    VALID_DIFFICULTIES,
    VALID_GAME_MODES,
    VALID_SECTION_KINDS,
)


def _section(**kw):
    defaults = dict(id="s1", kind="intro", length_beats=8, bpm=140, mode="cube",
                    intensity=0.5, pattern_refs=[], transitions=Transitions())
    defaults.update(kw)
    return Section(**defaults)


def test_level_plan_basic_construction():
    plan = LevelPlan(
        meta=LevelMeta(name="t", target_difficulty="medium", target_length_seconds=30.0),
        sections=[_section()],
    )
    d = plan.to_dict()
    assert d["meta"]["target_difficulty"] == "medium"
    assert len(d["sections"]) == 1


def test_level_plan_round_trip_via_dict():
    plan = LevelPlan(
        meta=LevelMeta(name="t"),
        sections=[_section(), _section(id="s2", kind="drop")],
    )
    raw = plan.to_dict()
    rebuilt = LevelPlan.from_dict(raw)
    assert rebuilt.sections[1].kind == "drop"


def test_section_validates_length_beats():
    with pytest.raises(ValueError):
        _section(length_beats=2)
    with pytest.raises(ValueError):
        _section(length_beats=128)


def test_section_validates_bpm():
    with pytest.raises(ValueError):
        _section(bpm=10)
    with pytest.raises(ValueError):
        _section(bpm=600)


def test_section_validates_intensity():
    with pytest.raises(ValueError):
        _section(intensity=1.5)
    with pytest.raises(ValueError):
        _section(intensity=-0.1)


def test_section_validates_kind_and_mode():
    with pytest.raises(ValueError):
        _section(kind="bogus")
    with pytest.raises(ValueError):
        _section(mode="dragon")


def test_level_plan_requires_at_least_one_section():
    with pytest.raises(ValueError):
        LevelPlan(meta=LevelMeta(name="t"), sections=[])


def test_json_schema_lists_match_constants():
    assert LEVEL_PLAN_JSON_SCHEMA["properties"]["sections"]["items"]["properties"]["mode"]["enum"] == list(VALID_GAME_MODES)
    assert LEVEL_PLAN_JSON_SCHEMA["properties"]["sections"]["items"]["properties"]["kind"]["enum"] == list(VALID_SECTION_KINDS)
    assert LEVEL_PLAN_JSON_SCHEMA["properties"]["meta"]["properties"]["target_difficulty"]["enum"] == list(VALID_DIFFICULTIES)


def test_level_plan_to_json_is_parseable():
    plan = LevelPlan(meta=LevelMeta(name="t"), sections=[_section()])
    payload = json.loads(plan.to_json())
    assert payload["sections"][0]["id"] == "s1"
