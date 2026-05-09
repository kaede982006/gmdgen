# GMD Generation Pipeline

The final `.gmd` file is produced by local deterministic code. Ollama can
change planning intent, but it cannot directly write the level.

## Pipeline Order

1. `UserPrompt`
2. `GenerationConfig`
3. `Ollama SectionPlan JSON`
4. `Local SectionIR`
5. `LevelIR`
6. `Group/Color Allocator`
7. `TriggerGraph`
8. `Serializer`
9. `SyntaxValidator`
10. `SemanticValidator`
11. `PlayabilityValidator`
12. `Repairer`
13. `Final GMD`
14. `GenerationReport`

## Local IR

`generate/ir.py` defines the contract types:

- `GenerationConfig`
- `SectionPlan`
- `LevelPlan`
- `LevelIR`
- `SectionIR`
- `GMDObjectIR`
- `TriggerIR`
- `GroupSymbol`
- `ColorSymbol`
- `GenerationArtifact`
- `ValidationResult`
- `RepairResult`
- `GenerationReport`

These types express the rule that symbolic planner intent must be decoded
before concrete ids exist.

## Allocation

`generate/allocator.py` assigns concrete group ids and color channel ids. This
is local deterministic behavior. Planner-provided numeric ids are invalid.

## Serialization

`generate/pipeline.py` demonstrates the IR serializer adapter:

- decode a symbolic plan to local IR
- allocate symbols
- convert IR to local `ObjectPlan` and `TriggerPlan`
- call the local serializer
- parse the serialized output and verify object count consistency

The production audio-conditioned path uses the existing rich local
`gd.plans.ObjectPlan` and `gd.plans.TriggerPlan` materializer as its runtime
local IR. Those types are not model output. Ollama output must first pass the
strict symbolic planner contract; direct model `object_plans` and
`trigger_plans` are rejected before they can reach the materializer.

## Final Success

A saved file is not enough. Final success requires:

- syntax validation passed
- semantic validation passed
- playability validation passed
- quality gate passed
- report consistency passed
- no planner fallback
- no low-quality draft state
