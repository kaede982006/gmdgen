from dataclasses import dataclass, field
from gmdgen.diagnostics.quality_failure import QualityFailure

@dataclass
class GenerationAuditReport:
    generation_id: str
    audio_file_name: str
    model: str
    raw_ai_plan_summary: dict = field(default_factory=dict)
    rendered_plan_summary: dict = field(default_factory=dict)
    description_quality_failures: list[QualityFailure] = field(default_factory=list)
    repair_loss_failures: list[QualityFailure] = field(default_factory=list)
    section_density_failures: list[QualityFailure] = field(default_factory=list)
    top_failure_reasons: list[str] = field(default_factory=list)

    def is_passed(self) -> bool:
        return len(self.top_failure_reasons) == 0

    def get_summary(self) -> dict:
        return {
            "generation_id": self.generation_id,
            "passed": self.is_passed(),
            "failures": self.top_failure_reasons,
            "raw_objects": self.raw_ai_plan_summary.get("object_count", 0),
            "rendered_objects": self.rendered_plan_summary.get("object_count", 0),
        }

class GenerationAuditor:
    @staticmethod
    def audit_generation(
        generation_id: str,
        audio_name: str,
        model: str,
        description: str,
        raw_objects: list[str],
        rendered_objects: list[str],
        sections: list[dict],
        drop_index: int = -1
    ) -> GenerationAuditReport:
        from gmdgen.diagnostics.output_inspector import OutputInspector
        report = GenerationAuditReport(
            generation_id=generation_id,
            audio_file_name=audio_name,
            model=model,
            raw_ai_plan_summary={"object_count": len(raw_objects)},
            rendered_plan_summary={"object_count": len(rendered_objects)},
        )

        desc_fails = OutputInspector.inspect_level_description(description)
        if desc_fails:
            report.description_quality_failures.extend(desc_fails)
            report.top_failure_reasons.extend([f.reason.value for f in desc_fails])

        loss_fails = OutputInspector.inspect_repair_loss(len(raw_objects), len(rendered_objects))
        if loss_fails:
            report.repair_loss_failures.extend(loss_fails)
            report.top_failure_reasons.extend([f.reason.value for f in loss_fails])

        density_fails = OutputInspector.inspect_section_density(sections, drop_index)
        if density_fails:
            report.section_density_failures.extend(density_fails)
            report.top_failure_reasons.extend([f.reason.value for f in density_fails])

        return report
