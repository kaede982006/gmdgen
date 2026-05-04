# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

def recommend_tuning(feedback_tags: list[str]) -> dict[str, Any]:
    recommendations = {}
    
    if "bad_drop" in feedback_tags:
        recommendations["drop_emphasis"] = {
            "action": "increase",
            "reason": "Users reported 'bad_drop'. Increasing drop emphasis.",
            "suggested_value": 1.5
        }
    if "too_empty" in feedback_tags:
        recommendations["object_budget"] = {
            "action": "increase",
            "reason": "Users reported 'too_empty'. Increasing object budget and minimum requirements.",
            "suggested_value": 1500
        }
        recommendations["min_final_object_count"] = {
            "action": "increase",
            "reason": "Ensuring minimum object count.",
            "suggested_value": 15
        }
    if "off_sync" in feedback_tags:
        recommendations["beat_snap_tolerance"] = {
            "action": "decrease",
            "reason": "Users reported 'off_sync'. Tightening beat snap tolerance.",
            "suggested_value": 0.04
        }
    if "no_decoration" in feedback_tags:
        recommendations["decoration_intensity"] = {
            "action": "increase",
            "reason": "Users reported 'no_decoration'. Increasing decoration intensity.",
            "suggested_value": 0.9
        }
    if "trigger_bad" in feedback_tags:
        recommendations["trigger_safety_level"] = {
            "action": "set",
            "reason": "Users reported 'trigger_bad'. Setting trigger safety level to 'safe'.",
            "suggested_value": "safe"
        }
        
    return recommendations
