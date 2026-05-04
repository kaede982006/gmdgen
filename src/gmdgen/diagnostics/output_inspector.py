from gmdgen.diagnostics.quality_failure import QualityFailureReason, QualityFailure
from gmdgen.diagnostics.string_sanitizer import MetadataSanitizer

class OutputInspector:
    @staticmethod
    def inspect_level_description(description: str) -> list[QualityFailure]:
        failures = []
        if MetadataSanitizer.detect_prompt_leak(description):
            failures.append(QualityFailure(QualityFailureReason.PROMPT_LEAK, "Prompt keywords detected in description."))
        if MetadataSanitizer.detect_json_blob(description):
            failures.append(QualityFailure(QualityFailureReason.JSON_BLOB_IN_DESCRIPTION, "JSON blob detected in description."))
        if MetadataSanitizer.detect_garbage_text(description):
            failures.append(QualityFailure(QualityFailureReason.GARBAGE_DESCRIPTION, "Description contains garbage text or is excessively long."))
        return failures

    @staticmethod
    def inspect_object_distribution(objects: list[str], min_count: int = 150) -> list[QualityFailure]:
        failures = []
        if len(objects) < min_count:
            failures.append(QualityFailure(QualityFailureReason.TOO_SPARSE, f"Only {len(objects)} objects found, expected at least {min_count}."))
        # Here we could extract block IDs and calculate diversity
        return failures

    @staticmethod
    def inspect_repair_loss(raw_count: int, final_count: int, max_loss_ratio: float = 0.3) -> list[QualityFailure]:
        failures = []
        if raw_count == 0:
            return failures
        loss_ratio = (raw_count - final_count) / raw_count
        if loss_ratio > max_loss_ratio:
            failures.append(QualityFailure(QualityFailureReason.HIGH_REPAIR_LOSS, f"Repair loss ratio {loss_ratio:.2f} exceeds {max_loss_ratio}."))
        return failures

    @staticmethod
    def inspect_section_density(sections: list[dict], drop_index: int = -1) -> list[QualityFailure]:
        failures = []
        if not sections:
            return failures
        # If there's a drop, it should have high density
        if drop_index != -1 and drop_index < len(sections):
            drop_density = sections[drop_index].get("object_count", 0)
            if drop_density < 20: # Example threshold
                failures.append(QualityFailure(QualityFailureReason.EMPTY_DROP, f"Drop section has only {drop_density} objects."))
        return failures
