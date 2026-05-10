# Quality Report — v2.3 Patches

## Failure mode this release fixes

The prior runtime produced "object spray" output:

| Symptom | Value |
|---|---|
| `raw_objects` | 0 |
| `final_objects` | 46258 |
| `triggers` | 0 |
| Visual | no ground line, no path, no structure |
| Save gate | candidate `selected=False` but file still saved |

Root cause: the AI was asked to emit every object directly. With token
limits the response was empty; a repair loop then "filled" the empty
canvas with 46k random objects.

## v2.3 invariant gates

The save path is now protected by five structural invariants
(`gmdgen.generate.invariants`). Any candidate that fails *any* gate is
**rejected**, never repaired into existence.

| Invariant | Threshold | Spray case | New synthetic baseline |
|---|---|---|---|
| I-1 object count | `[50, 12000]` | 46258 → **abort** | 200–600 → pass |
| I-2 trigger floor | `≥ max(3, sections × 1)` | 0 → **abort** | 5+ via `insert_triggers` |
| I-3 ground coverage | `≥ 0.70` | 0 → **abort** | 1.0 (rail by construction) |
| I-4 jumpable path | `≥ 0.95` | n/a | ≥ 0.95 with new patterns |
| I-5 unique types | `≥ 6` | n/a | ≥ 6 (palette rotation) |
| R0 `raw_objects ≠ 0` | hard reject | 0 → **abort** | enforced |

## Verification

`tests/test_no_object_spray.py` exercises every attack vector that
produced the spray failure:

| Test | Attack | Outcome |
|---|---|---|
| `test_empty_raw_objects_aborts_save` | empty AI response | `R0` raises |
| `test_oversized_output_aborts` | 12 100 objects | `I-1` raises |
| `test_zero_triggers_aborts` | trigger pipeline broken | `I-2` raises |
| `test_single_id_flood_aborts` | one object ID 500× | `I-5` raises |
| `test_no_ground_coverage_aborts` | skywriting | `I-3` raises |
| `test_unjumpable_path_aborts` | hostile patterns | `I-4` raises |
| `test_object_count_band_gating` | parametrized 0/10/200/15k/50k | only 200 passes |

## New architecture (HSR — Hierarchical Structural Representation)

```
RawSpec
  └─ AI(plan)        : LevelPlan  (~few hundred tokens, JSON-schema enforced)
  └─ expand_plan     : ObjectGraph (deterministic)
  └─ insert_triggers : adds I-2 floor by construction
  └─ simulate_play   : computes jumpable_path_ratio
  └─ assert_invariants: I-1..I-5 + R0
  └─ encode          : .gmd (only the selected candidate is encoded)
```

The AI is **never** used for bulk object generation. Same `(plan, seed)`
yields byte-identical output (`tests/test_hsr_expand.py::test_expand_is_deterministic_with_seed`).

## Pattern library

| Mode | Easy | Medium | Hard | Total |
|---|---|---|---|---|
| cube | 6 | 6 | 6 | 18 |
| ship | 6 | 6 | 6 | 18 |
| ball | 6 | 6 | 6 | 18 |
| ufo | 6 | 6 | 6 | 18 |
| wave | 6 | 6 | 6 | 18 |
| robot | 6 | 6 | 6 | 18 |
| spider | 6 | 6 | 6 | 18 |
| **Total** | **42** | **42** | **42** | **126** |

Every cell of `mode × difficulty` is populated. The synthesizer is
deterministic; `tests/test_pattern_library.py` enforces:

- 21 cells filled (no empty cell)
- ≥ 6 patterns per cell
- ≥ 100 patterns total
- All required schema fields present
- `pick_pattern` deterministic with seed

## Assumptions

- Synthetic patterns are a defensible default. Hand-tuned patterns can
  later be added without changing the loader contract.
- The play solver is heuristic, not a full GD physics simulator.

## Constraints

- No live Gemini in pytest.
- No bulk object generation through AI.
- No Gemini/OpenAI/Claude runtime providers.

## Risks

- Pattern library is synthetic; some patterns may feel mechanical.
  Users with their own dataset can populate the learning store and the
  pattern selector will use those as priors via `build_palette_from_learned_store`.
- `play_solver` uses heuristic limits (MIN_REACTION_X = 60u, MAX_JUMP_Y = 90u).
  These are conservative; real-game tolerances may differ.
