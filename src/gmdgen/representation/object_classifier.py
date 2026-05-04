"""Rule-based GD object classifier.

Goodfellow Ch.1 §1.0 "feature engineering":
  Each object ID is classified into one of five semantic classes so that the
  tokenizer and trainer can treat them as *different factors of variation*
  rather than exchangeable integers.

Classes
-------
structure    — solid/hazard blocks that define level geometry
decoration   — purely visual objects (no gameplay effect)
trigger      — activates effects on groups (color, move, alpha, …)
portal       — changes player state (speed, gamemode, gravity, mirror)
special      — coins, checkpoints, orbs, pads, rings
unknown      — anything not in the known sets (handled gracefully)
"""

from __future__ import annotations

from enum import Enum


class ObjectClass(str, Enum):
    STRUCTURE = "structure"
    DECORATION = "decoration"
    TRIGGER = "trigger"
    PORTAL = "portal"
    SPECIAL = "special"
    UNKNOWN = "unknown"


# ─────────────────────────────────────────────────────────────
# Known ID sets — sourced from GD object/prop documentation
# IDs are stored as strings to match the tokeniser representation.
# ─────────────────────────────────────────────────────────────

# Speed portals
_SPEED_PORTAL_IDS: frozenset[str] = frozenset(
    {"200", "201", "202", "203", "1334"}
)

# Gamemode portals (cube/ship/ball/ufo/wave/robot/spider/swing)
_GAMEMODE_PORTAL_IDS: frozenset[str] = frozenset(
    {"12", "13", "47", "111", "660", "745", "1331"}
)

# Mirror / gravity / dual / mini portals
_STATE_PORTAL_IDS: frozenset[str] = frozenset(
    {"45", "46", "99", "101", "286", "287", "1163", "1164", "1231", "1705"}
)

_PORTAL_IDS: frozenset[str] = (
    _SPEED_PORTAL_IDS | _GAMEMODE_PORTAL_IDS | _STATE_PORTAL_IDS
)

# Triggers (color, move, alpha, toggle, spawn, pulse, shake, rotate, follow …)
_TRIGGER_IDS: frozenset[str] = frozenset(
    {
        "29",   "30",   "32",   "33",   "105",  "142",  "200",  "228",
        "229",  "338",  "660",  "899",  "901",  "915",  "1006", "1007",
        "1049", "1268", "1346", "1347", "1520", "1585", "1616", "2067",
        "2901", "3016", "3017", "3018", "3019", "3020", "3021", "3022",
        "3023", "3024", "3025", "3029", "3030", "3031", "3032", "3033",
        "3034", "3607", "3608", "3609", "3612", "3613", "3614", "3615",
        "3617", "3618", "3619", "3620", "3641", "3642",
    }
)

# Orbs / pads (interactive jump mechanics)
_ORB_PAD_IDS: frozenset[str] = frozenset(
    {
        "35",  "36",  "84",  "141", "143", "1022", "1330", "1704", "1751",
        "10"   ,  "11",  "67",  "140",
    }
)

# Coins, checkpoints, end-portals
_SPECIAL_IDS: frozenset[str] = frozenset(
    {
        "142",  # not trigger in all contexts — kept here too
        "1329", # checkpoint
        "1594", # end trigger
        "1275", # collectible coin
        "142",  # spawn
        "1234", "1595",
    }
) | _ORB_PAD_IDS

# Solid structure blocks (approximate ranges + known IDs)
# GD v2.2: block IDs roughly 1–699 contain both structure and deco;
# we whitelist known solid types here.
_STRUCTURE_IDS: frozenset[str] = frozenset(
    {str(i) for i in range(1, 8)}          # classic blocks 1-7
    | {str(i) for i in range(39, 56)}       # spikes / hazards
    | {str(i) for i in range(57, 84)}       # extra basic blocks
    | {str(i) for i in range(85, 100)}
    | {str(i) for i in range(103, 115)}
    | {"130", "131", "132", "133", "134",
       "211", "216", "293", "305", "338"}   # commonly used solids
)

# Decoration: anything not in the above categories and ID ≤ 1750 is deco
_MAX_KNOWN_DECO_ID = 1750


def classify(object_id: str) -> ObjectClass:
    """Return the ObjectClass for a given GD object ID string."""
    if object_id in _PORTAL_IDS:
        return ObjectClass.PORTAL
    if object_id in _TRIGGER_IDS:
        return ObjectClass.TRIGGER
    if object_id in _SPECIAL_IDS:
        return ObjectClass.SPECIAL
    if object_id in _STRUCTURE_IDS:
        return ObjectClass.STRUCTURE

    if not object_id.isdigit():
        return ObjectClass.UNKNOWN

    id_int = int(object_id)
    if id_int <= 0:
        return ObjectClass.UNKNOWN
    if id_int <= _MAX_KNOWN_DECO_ID:
        return ObjectClass.DECORATION

    return ObjectClass.UNKNOWN


def is_structural(object_id: str) -> bool:
    cls = classify(object_id)
    return cls in (ObjectClass.STRUCTURE, ObjectClass.PORTAL, ObjectClass.SPECIAL)


def is_visible(object_id: str) -> bool:
    cls = classify(object_id)
    return cls in (
        ObjectClass.STRUCTURE,
        ObjectClass.DECORATION,
        ObjectClass.SPECIAL,
        ObjectClass.PORTAL,
    )


def class_short(object_id: str) -> str:
    """Return a single-character class abbreviation for use in tokens."""
    _MAP = {
        ObjectClass.STRUCTURE: "S",
        ObjectClass.DECORATION: "D",
        ObjectClass.TRIGGER: "T",
        ObjectClass.PORTAL: "P",
        ObjectClass.SPECIAL: "X",
        ObjectClass.UNKNOWN: "U",
    }
    return _MAP[classify(object_id)]
