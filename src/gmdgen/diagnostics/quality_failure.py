from enum import Enum
from dataclasses import dataclass

class QualityFailureReason(str, Enum):
    GARBAGE_DESCRIPTION = "garbage_description"
    PROMPT_LEAK = "prompt_leak"
    JSON_BLOB_IN_DESCRIPTION = "json_blob_in_description"
    TOO_SPARSE = "too_sparse"
    EMPTY_DROP = "empty_drop"
    HIGH_REPAIR_LOSS = "high_repair_loss"
    LOW_OBJECT_DIVERSITY = "low_object_diversity"
    POOR_MUSIC_SYNC = "poor_music_sync"
    FATAL_VALIDATION_ERROR = "fatal_validation_error"
    GEODE_REJECTION = "geode_rejection"
    BAD_LEARNING_MEMORY = "bad_learning_memory"

@dataclass
class QualityFailure:
    reason: QualityFailureReason
    details: str
    severity: str = "error"  # 'warning' or 'error'
