# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from gmdgen.gd.plans import ObjectPlan, SectionPlan, TriggerPlan
from gmdgen.gd.time_mapping import SpeedState
from gmdgen.generate.quality import (
    build_quality_feedback_prompt,
    build_repair_quality_report,
    compute_actual_density_by_section,
    compute_density_target_by_section,
    compute_density_target_error,
    compute_drop_impact_score,
    diff_snapshots,
    snapshot_from_plans,
    summarize_quality_loss,
    CandidateReport,
)


def _sections() -> list[SectionPlan]:
    return [
        SectionPlan(0, 1, 0, 300, "intro", "cube", SpeedState.NORMAL, 0.2, 0.2, 0.1, 0.2),
        SectionPlan(1, 2, 300, 600, "buildup", "cube", SpeedState.NORMAL, 0.5, 0.4, 0.4, 0.5),
        SectionPlan(2, 3, 600, 900, "drop", "cube", SpeedState.NORMAL, 0.9, 0.8, 0.8, 0.8),
    ]


def test_plan_snapshot_records_all_stages() -> None:
    snapshot = snapshot_from_plans(
        "raw_ai_plan",
        [ObjectPlan("1", 100, 90, "ai_structure")],
        [TriggerPlan("pulse", "1006", 100, 300, target_group=1)],
        _sections(),
    )

    assert snapshot.stage == "raw_ai_plan"
    assert snapshot.object_count == 1
    assert snapshot.trigger_count == 1
    assert snapshot.role_distribution["ai_structure"] == 1


def test_plan_diff_counts_removed_objects_and_triggers() -> None:
    before = snapshot_from_plans(
        "normalized_plan",
        [ObjectPlan("1", 100, 90, "ai_structure"), ObjectPlan("500", 140, 240, "safe_decoration")],
        [TriggerPlan("pulse", "1006", 100, 300, target_group=1)],
        _sections(),
    )
    after = snapshot_from_plans(
        "repaired_plan",
        [ObjectPlan("1", 100, 90, "ai_structure")],
        [],
        _sections(),
    )

    diff = diff_snapshots(before, after)

    assert diff.removed_objects == 1
    assert diff.removed_triggers == 1


def test_repair_quality_report_records_pruning_reasons() -> None:
    report = build_repair_quality_report(
        plan_validation_warnings=[
            "removed_unsupported_trigger: shake",
            "removed_orphan_trigger: pulse target=9",
            "object_budget_pruned: removed 2",
        ]
    )

    assert report.removed_due_to_unsupported_trigger == 1
    assert report.removed_due_to_missing_target_group == 1
    assert report.removed_due_to_object_budget == 1


def test_quality_loss_summary_identifies_major_cause() -> None:
    report = build_repair_quality_report(plan_validation_warnings=["object_budget_pruned: removed 20"])
    report.pruned_irrelevant_trigger_properties = 4

    reasons = summarize_quality_loss(
        report,
        raw_object_count=100,
        final_object_count=40,
        raw_trigger_count=20,
        final_trigger_count=5,
        drop_impact_score=0.2,
    )

    assert any("Trigger properties were pruned" in reason for reason in reasons)
    assert any("sparse" in reason for reason in reasons)
    assert any("Drop section lost impact" in reason for reason in reasons)


def test_density_target_reported_and_drop_impact_score() -> None:
    sections = _sections()
    objects = [
        ObjectPlan("1", 100, 90, "ai_structure"),
        ObjectPlan("1", 650, 90, "ai_structure"),
        ObjectPlan("36", 700, 180, "beat_orb"),
        ObjectPlan("500", 720, 240, "visual_accent_target"),
    ]
    triggers = [TriggerPlan("pulse", "1006", 720, 300, target_group=1)]
    target = compute_density_target_by_section(sections)
    actual = compute_actual_density_by_section(objects, triggers, sections)

    assert set(target) == {"0", "1", "2"}
    assert compute_density_target_error(target, actual) >= 0.0
    assert compute_drop_impact_score(sections, actual, {"2": 1}) > 0.0


def test_retry_prompt_mentions_repair_loss_and_empty_drop() -> None:
    report = CandidateReport(candidate_id=1, object_count=2, trigger_count=0, drop_impact=0.1, reject_reason="too_many_invalid_triggers")

    prompt = build_quality_feedback_prompt(report, ["Final map is sparse because pruning removed 48% of objects."])

    assert "sparse" in prompt
    assert "drop_too_weak" in prompt
    assert "too_many_invalid_triggers" in prompt
    assert "sk-" not in prompt
