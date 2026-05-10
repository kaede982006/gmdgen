# Algorithmic Audit

This audit records the current generation architecture after the planner/IR
contract cleanup. It is intentionally about algorithmic responsibility, not
release operations.

## Current Role Split

- Gemini is a bounded planner. `GeminiProvider.generate_level_plan` accepts
  strict `level_plan` + `sections` JSON and rejects raw `.gmd`, legacy
  `object_plans`, concrete `group_id`, concrete `color_channel_id`, final
  scores, and validation verdicts.
- Local deterministic code owns final generation. The audio-conditioned path
  builds deterministic object plans, allocates groups, validates trigger
  targets, serializes through `plans_to_level_objects`, validates the saved
  `.gmd`, computes quality metrics, and writes a `ValidationReport`.
- The source-of-truth adapter lives in `generate/ir.py`, `generate/decoder.py`,
  `generate/allocator.py`, and `generate/pipeline.py`. The production generator
  still uses the richer existing `gd.plans` materialization types, but the
  planner contract is now represented as symbolic IR before serialization.

## Raw String Dependency

Raw `.gmd` strings are still present at serializer and file IO boundaries:

- `plans_to_level_objects` emits Geometry Dash object strings from local plans.
- `_compose_level_data`, `encode_level_data`, and `save_level_output` write the
  final level data.
- `validate_gmd_file`, `round_trip_validate`, and `validate_save_string_safety`
  parse serialized output for syntax and editor safety.

These raw strings are local artifacts only. They are not valid Gemini output
and are rejected by `ai/planner.py` and `ai/Gemini_provider.py`.

## Source of Truth

The intended order is:

`UserPrompt` -> `GenerationConfig` -> `Gemini SectionPlan JSON` ->
`Local SectionIR` -> `LevelIR` -> `Group/Color Allocator` ->
`TriggerGraph` -> `Serializer` -> `SyntaxValidator` ->
`SemanticValidator` -> `PlayabilityValidator` -> `Repairer` ->
`Final GMD` -> `GenerationReport`.

Gemini is allowed only up to `GenerationConfig` and `SectionPlan`. Concrete
group ids, color channels, trigger target ids, final scoring, playability
verdicts, report consistency, and serialization are local responsibilities.

## Legacy Compatibility Adapter

`AILevelPlanResponse` still exists for migration tests and non-Gemini internal
adapters. Its direct `object_plans` and `trigger_plans` fields are deprecated
compatibility fields, not production Gemini output and not report source of
truth. The Gemini provider and production audio path reject model-provided
object plans before they can reach the materializer.

## Fallback Risk

The previous GUI wording could make a saved fallback look like success because
file write success and generation success were conflated. The report now
separates:

- `planner_status`
- `planner_fallback_used`
- `quality_gate_passed`
- `low_quality_draft_saved`
- `final_success`

`final_success` is false when the planner fallback was used or when a
low-quality draft is saved.

## Count Consistency Risk

The report consistency gate checks candidate, parsed, serialized, and final
counts when those fields are present. It hard-fails count mismatches, zero final
objects, missing target groups, missing color channels, invalid trigger
targets, stale repair metrics, low-quality draft marked as success, fallback
marked as final success, and undefined candidate/final score definitions.

## Pattern Schema Boundary

`patterns_index.json` is an index file, not an individual pattern file.
`validation/pattern_schema.py` now validates index payloads and object pattern
payloads with separate rules. The library report counts index files separately
from valid pattern files.

## GUI Status Boundary

The GUI now maps generation reports into one of these states:

- `final_success`: syntax, semantic, playability, quality, and report checks
  passed without planner fallback.
- `fallback_draft`: Gemini failed or returned invalid planner output; a
  deterministic draft exists for inspection only.
- `low_quality_draft`: output was saved, but validation or the quality gate
  failed.
- `incomplete`: output exists but no final success contract was recorded.

Tracebacks are kept out of user dialogs; sanitized detail remains in logs and
reports.

## Optional Tool Handling

Code validation treats optional static tools such as `ruff`, `mypy`, and
`pyright` as warning/skip when not installed. `compileall` and import failures
remain hard failures because they indicate broken runtime code.
