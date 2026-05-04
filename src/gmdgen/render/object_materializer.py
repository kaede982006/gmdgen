from __future__ import annotations

import random
from typing import Any

from gmdgen.gd.plans import ObjectPlan


def materialize_object_roles(
    object_plans: list[ObjectPlan],
    *,
    style_summary: dict[str, Any],
    seed: int = 42,
) -> list[ObjectPlan]:
    rng = random.Random(seed)
    
    # Fallback default IDs for common roles if style_summary doesn't have them
    default_orb_ids = ["36", "84", "141", "1022"]
    default_pad_ids = ["35", "140"]
    default_struct_ids = ["1", "2", "3", "4", "5", "8"]
    default_deco_ids = ["500", "501", "502", "503"]
    
    orb_ids = style_summary.get("object_classes", {}).get("orbs", default_orb_ids) or default_orb_ids
    pad_ids = style_summary.get("object_classes", {}).get("pads", default_pad_ids) or default_pad_ids
    struct_ids = style_summary.get("object_classes", {}).get("structures", default_struct_ids) or default_struct_ids
    deco_ids = style_summary.get("object_classes", {}).get("decorations", default_deco_ids) or default_deco_ids
    
    materialized = []
    
    for plan in object_plans:
        role = str(plan.role).lower()
        if role == "beat_orb" or role == "rhythm_orb":
            plan.object_id = str(rng.choice(orb_ids))
        elif role == "beat_pad":
            plan.object_id = str(rng.choice(pad_ids))
        elif role == "ai_structure" or role == "structure":
            plan.object_id = str(rng.choice(struct_ids))
        elif role == "safe_decoration" or role == "visual_accent_target":
            plan.object_id = str(rng.choice(deco_ids))
        elif role == "custom":
            pass # Keep as is
            
        # Ensure we always have some valid object_id
        if not plan.object_id or plan.object_id == "0":
            plan.object_id = "1"
            
        materialized.append(plan)
        
    return materialized
