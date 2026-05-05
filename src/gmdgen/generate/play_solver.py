# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 kaede982006
"""Lightweight playability solver.

This is **not** a full Geometry Dash physics simulator; it is a fast
heuristic that flags structural impossibilities a player would always hit:

1. Two consecutive obstacles closer than the minimum reaction window.
2. Vertical gaps that exceed the cube's max-jump height (a hazard ID
   sitting above where the player can reach without an orb).

The solver returns a ``PlayReport`` whose ``success`` flag is the gate for
downstream candidate selection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Heuristic limits for the cube — sufficient to catch obvious failures.
MAX_JUMP_X = 280.0  # horizontal jump distance at speed=1
MAX_JUMP_Y = 90.0   # vertical reach without an orb
MIN_REACTION_X = 60.0  # min horizontal gap between two consecutive hazards
GROUND_Y = 105


@dataclass(slots=True)
class PlayReport:
    success: bool
    coverage: float = 0.0
    failed_at_x: float | None = None
    reason: str = ""
    jumpable_path_ratio: float = 1.0
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "coverage": self.coverage,
            "failed_at_x": self.failed_at_x,
            "reason": self.reason,
            "jumpable_path_ratio": self.jumpable_path_ratio,
            "issues": list(self.issues),
        }


def _is_hazard(obj: Any) -> bool:
    role = getattr(obj, "role", "")
    if isinstance(role, str) and role.lower() in {"gameplay", "hazard"}:
        obj_id = str(getattr(obj, "object_id", ""))
        # spike-class IDs in gameplay objects.
        return obj_id in {"8", "9", "39", "103"}
    return False


def simulate_play(objects: list[Any]) -> PlayReport:
    """Run a fast structural playability check over a sorted object list."""
    if not objects:
        return PlayReport(success=False, reason="empty_object_list")

    # Sort defensively; expander already produces monotone x.
    items = sorted(objects, key=lambda o: getattr(o, "x", 0.0))

    issues: list[str] = []
    last_hazard_x: float | None = None
    failed_at: float | None = None
    total_pairs = 0
    bad_pairs = 0

    for obj in items:
        x = float(getattr(obj, "x", 0.0))
        y = float(getattr(obj, "y", GROUND_Y))

        if _is_hazard(obj):
            # Reaction-window check.
            if last_hazard_x is not None:
                total_pairs += 1
                gap = x - last_hazard_x
                if gap < MIN_REACTION_X:
                    bad_pairs += 1
                    issues.append(f"tight_hazard_pair@{x:.1f} gap={gap:.1f}")
                    if failed_at is None:
                        failed_at = x
            last_hazard_x = x

            # Reach check (only the rough "out of jump range" case).
            if y - GROUND_Y > MAX_JUMP_Y * 1.5:
                issues.append(f"hazard_out_of_reach@{x:.1f} y={y:.1f}")
                if failed_at is None:
                    failed_at = x

    # Heuristic coverage: x extent / approximate level length.
    x_min = min((float(getattr(o, "x", 0.0)) for o in items), default=0.0)
    x_max = max((float(getattr(o, "x", 0.0)) for o in items), default=0.0)
    coverage = 1.0 if x_max > x_min else 0.0

    jumpable = 1.0 if total_pairs == 0 else max(0.0, 1.0 - bad_pairs / total_pairs)
    success = jumpable >= 0.95 and failed_at is None and not any("out_of_reach" in i for i in issues)

    return PlayReport(
        success=success,
        coverage=coverage,
        failed_at_x=failed_at,
        reason="" if success else (issues[0] if issues else "structural_failure"),
        jumpable_path_ratio=jumpable,
        issues=issues[:20],
    )
