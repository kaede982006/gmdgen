# Ollama Planner Architecture

Ollama is a symbolic planner. It describes intent; it does not generate final
Geometry Dash save data.

## Allowed Ollama Responsibilities

- Convert user natural language into planning intent.
- Produce `level_plan` and symbolic `sections`.
- Describe style, difficulty, density, sync intent, and readable design notes.
- Interpret validator reports for a future repair strategy.
- Suggest repair strategy in natural language.
- Summarize reports for the GUI.

## Forbidden Ollama Responsibilities

- Raw `.gmd` or save-string output.
- Direct object strings such as `1,100,2,30,3,...`.
- Concrete `group_id` or `group_ids`.
- Concrete `color_channel_id` or `color_channel`.
- Concrete trigger target ids.
- Final scores.
- Playability verdicts.
- Serializer behavior.
- Validator behavior.
- GenerationReport source of truth.

## Strict JSON Contract

Ollama output must be a JSON object with exactly these top-level keys:

```json
{
  "level_plan": {
    "level_name": "string",
    "difficulty": "easy|normal|hard|insane|demon",
    "target_duration": 30.0,
    "object_budget": 500,
    "style": "modern_glow",
    "sync_intensity": "low|medium|high"
  },
  "sections": [
    {
      "section_id": "s001",
      "time_start": 0.0,
      "time_end": 8.0,
      "game_mode": "cube|ship|ball|ufo|wave|robot|spider",
      "speed": "0.5x|1x|2x|3x|4x",
      "density": 0.35,
      "primary_pattern": "intro_platforming",
      "allowed_object_families": ["block", "spike", "orb", "pad"],
      "forbidden_features": ["unbounded_trigger_spam"],
      "trigger_budget": 3,
      "group_symbols": ["intro_blocks"],
      "design_notes": "short readable intro"
    }
  ]
}
```

Concrete ids are replaced with symbols. A valid planner target is
`target_group_symbol: "pulse_block_A"`, not `target_group: 17`.

## Failure Handling

Invalid planner output is rejected with a precise error:

- JSON parse failure
- unknown enum
- schema mismatch
- over-budget section
- forbidden concrete id field
- raw `.gmd` string

The fallback planner returns a deterministic template plan and records
`planner_fallback_used=true`. Fallback is degraded mode; it is not an AI quality
success.
