# Generation Report Contract

GenerationReport is the release gate for generated output. It must describe
what actually happened, not what the GUI hoped happened.

## Required Fields

- `planner_status`
- `planner_fallback_used`
- `candidate_ir_objects`
- `serialized_objects`
- `final_objects`
- `syntax_validation`
- `semantic_validation`
- `playability_validation`
- `repair_applied`
- `repair_loss`
- `quality_gate_passed`
- `low_quality_draft_saved`
- `final_success`

## Hard Failures

The report consistency gate fails when:

- `raw_objects == 0` while candidate or final objects exist, when `raw_objects`
  is explicitly reported
- `candidate_objects != parsed_candidate_objects`
- `final_objects != serialized_objects`
- `final_objects == 0`
- `missing_target_group > 0`
- `missing_color_channel > 0`
- `invalid_trigger_target > 0`
- repair was applied but repair metrics were not updated
- `low_quality_draft_saved=true` and `final_success=true`
- `planner_fallback_used=true` and `final_success=true`
- candidate or final score definition is explicitly missing

## Draft Semantics

`low_quality_draft_saved=true` is allowed only when the output is not final.
The normal expected pairing is:

```json
{
  "low_quality_draft_saved": true,
  "quality_gate_passed": false,
  "final_success": false
}
```

Gemini fallback follows the same rule:

```json
{
  "planner_fallback_used": true,
  "final_success": false
}
```

Fallback proves runtime resilience, not AI generation quality.

## Code Gate

`validation/report_consistency.py` exposes
`validate_generation_report_consistency(report, hard=True)`. Tests cover count
mismatch, missing target group, low-quality draft success confusion, planner
fallback success confusion, and stale repair metrics.
