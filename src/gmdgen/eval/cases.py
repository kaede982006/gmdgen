from dataclasses import dataclass, field
from typing import Any

@dataclass
class EvalCase:
    case_id: str
    audio_fixture: str
    reference_level_fixture: str = ""
    difficulty: str = "normal"
    target_style: str = ""
    quality_mode: str = "Balanced"
    expected_min_object_count: int = 10
    expected_min_trigger_count: int = 0
    expected_max_repair_loss: float = 0.5
    expected_min_drop_impact: float = 0.0
    expected_min_density_alignment: float = 0.0
    expected_min_object_diversity: float = 0.0
    expected_min_beat_sync: float = 0.0
    expected_max_playability_warnings: int = 10
    expected_min_editor_safety: float = 0.5
    expected_tags: list[str] = field(default_factory=list)
