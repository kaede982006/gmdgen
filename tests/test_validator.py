# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from gmdgen.generate.validator import validate_gmd_text
from gmdgen.io.gmd_decoder import encode_level_data


def test_validator_accepts_valid_text() -> None:
    encoded = encode_level_data("kA11,0;1,1,2,30,3,180;")
    raw = f"<d><k>k2</k><s>ok</s><k>k4</k><s>{encoded}</s></d>"

    is_valid, issues = validate_gmd_text(raw)
    assert is_valid is True
    assert issues == []


def test_validator_reports_orphan_trigger_and_budget() -> None:
    encoded = encode_level_data(
        "kA11,0;"
        "1,1,2,30,3,180;"
        "1,1006,2,60,3,180,51,99;"
    )
    raw = f"<d><k>k2</k><s>bad</s><k>k4</k><s>{encoded}</s><k>k95</k><i>2</i></d>"

    is_valid, issues = validate_gmd_text(raw, object_budget=1)

    assert is_valid is False
    assert any("orphan_trigger" in issue for issue in issues)
    assert any("object_budget_exceeded" in issue for issue in issues)
