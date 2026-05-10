# Algorithmic Background

gmdgen treats level generation as a data and validation pipeline. Model calls
are optional planning inputs, not the source of truth.

## Core Principles

1. Gemini 7B is a controlled section planner, not a `.gmd` generator.
2. Final `.gmd` creation belongs to the local deterministic pipeline.
3. The ML starting point is clean data, typed IR, validation, and repair.
4. Raw `.gmd` generation from the model is forbidden.
5. GUI success means final validation and report consistency passed.
6. Gemini fallback proves only that the app did not crash; it does not prove AI
   generation quality.

## Dirty Data Assumption

Every external `.gmd` or pattern payload is treated as dirty until validated.

Incomplete data:

- missing object properties
- missing group references
- missing color channels
- missing trigger targets
- incomplete metadata

Noisy data:

- abnormal coordinates
- excessive object density
- broken trigger properties
- editor artifacts
- duplicate decoration objects

Inconsistent data:

- raw, candidate, final, and report object counts disagree
- triggers reference groups that are not defined
- report score and validation result disagree
- repair metrics are not refreshed after repair
- low-quality draft is presented as final success

## Source-of-Truth Pipeline

The enforced order is:

`UserPrompt` -> `GenerationConfig` -> `Gemini SectionPlan JSON` ->
`Local SectionIR` -> `LevelIR` -> `Group/Color Allocator` ->
`TriggerGraph` -> `Serializer` -> `SyntaxValidator` ->
`SemanticValidator` -> `PlayabilityValidator` -> `Repairer` ->
`Final GMD` -> `GenerationReport`.

Only `GenerationConfig` and `SectionPlan` may come from Gemini. Everything
after that is deterministic and locally checked.

## Code Anchors

- Planner contract: `src/gmdgen/ai/planner.py`
- Strict Gemini provider boundary: `src/gmdgen/ai/Gemini_provider.py`
- IR dataclasses: `src/gmdgen/generate/ir.py`
- SectionPlan to IR decoder: `src/gmdgen/generate/decoder.py`
- Symbol allocator: `src/gmdgen/generate/allocator.py`
- Serializer adapter: `src/gmdgen/generate/pipeline.py`
- Report consistency gate: `src/gmdgen/validation/report_consistency.py`
- Pattern schema validation: `src/gmdgen/validation/pattern_schema.py`
- Dataset cleaning report: `src/gmdgen/validation/data_cleaning.py`
