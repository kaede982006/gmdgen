# Data Cleaning and Pattern Schema

The pattern library and user datasets are validation inputs, not trusted truth.

## Pattern Files

Pattern object JSON and `patterns_index.json` use different schemas.

Pattern object files require:

- `id`
- `mode`
- `difficulty`
- `length_beats`
- `objects`
- `entry`
- `exit`
- `tested`
- `source`

Each object is checked for:

- required fields
- object count greater than zero
- `x_beat`, `y`, `dx`, and `dy` numeric ranges
- mode and difficulty enums
- supported trigger type when an object is marked as a trigger
- symbolic group/color reference integrity

`patterns_index.json` is validated as an index with `cells` and `patterns`. It
is not counted as an individual pattern object file.

## Library Report Shape

```json
{
  "checked_files": 127,
  "valid_patterns": 126,
  "index_files": 1,
  "invalid_patterns": [],
  "warnings": [],
  "repair_suggestions": [],
  "destructive_changes": false
}
```

## Dataset Cleaning

User `.gmd` datasets are never automatically deleted or rewritten. The cleaning
path is report-first:

- parse file
- decode `k4`
- count objects
- compare `k95` when present
- report invalid files and repair suggestions

The result records `destructive_changes=false`.

## Code Anchors

- `validation/pattern_schema.py`
- `validation/data_cleaning.py`
- `tests/test_pattern_schema_validation.py`

## Regeneration

Default tests are read-only. To intentionally regenerate package pattern
fixtures, use one of these maintenance commands:

```bash
GMDGEN_REGENERATE_PATTERN_FIXTURES=1 python -m gmdgen.patterns.regenerate
GMDGEN_REGENERATE_PATTERN_FIXTURES=1 python scripts/regenerate_pattern_fixtures.py
```

Do not run fixture regeneration as part of normal validation or release tests.
