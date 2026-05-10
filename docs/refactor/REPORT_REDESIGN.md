# Phase G — Algorithmic Redesign Report (port-based)

_Generated automatically. Source of truth: `wip/session-snapshot-20260504-202428` branch in gmdgen-legacy._

## Scope

Phase G is fulfilled by porting the wip session-snapshot tree as the initial 0.1.0 commit. No fresh redesign was executed in this overhaul; the redesign work had already been completed across multiple prior sessions and committed to the wip snapshot.

## origin/main vs wip — diff --stat

```
 tests/test_gui_Gemini_only_surface.py              |  70 ++-
 tests/test_gui_quality_messages.py                 | 114 ++++
 tests/test_gui_worker.py                           |   8 +-
 tests/test_learning_feature_extractor.py           |   4 +-
 tests/test_learning_palette_fallback.py            | 109 ++++
 tests/test_learning_store.py                       |   8 +-
 tests/test_marker_spam_prevention.py               | 135 +++++
 .../test_materializer_playability_preservation.py  | 117 ++++
 tests/test_mode_transition_planning.py             | 113 ++++
 tests/test_no_improvement_early_stopping.py        |  52 ++
 tests/test_object_count_metadata_consistency.py    | 120 ++++
 tests/test_object_count_scaling.py                 |  96 +++
 tests/test_object_diversity.py                     |  80 +++
 tests/test_Gemini_json_recovery.py                 | 140 +++++
 tests/test_playable_skeleton_generation.py         | 118 ++++
 tests/test_preference_export.py                    |   4 +-
 tests/test_quality_draft_diagnostics.py            | 121 ++++
 tests/test_repair_loss_guard.py                    |  95 +++
 tests/test_report_text_sanity.py                   | 121 ++++
 55 files changed, 3554 insertions(+), 565 deletions(-)
```

## Changed modules

- .gitignore
- fix_gui_environment_key.py
- src/gmdgen/ai/cache.py
- src/gmdgen/ai/factory.py
- src/gmdgen/ai/fine_tune_export.py
- src/gmdgen/ai/Gemini_provider.py
- src/gmdgen/ai/preference_export.py
- src/gmdgen/errors.py
- src/gmdgen/eval/__init__.py
- src/gmdgen/eval/critic.py
- src/gmdgen/eval/live_Gemini_eval.py
- src/gmdgen/feedback/store.py
- src/gmdgen/gd/plans.py
- src/gmdgen/generate/audio_conditioned.py
- src/gmdgen/generate/materializer.py
- src/gmdgen/generate/quality_gate.py
- src/gmdgen/generate/scoring.py
- src/gmdgen/generate/validator.py
- src/gmdgen/gui/app.py
- src/gmdgen/learning/feature_extractor.py
- src/gmdgen/learning/store.py
- temp_refactor.py

## New tests (snapshot)

- tests/test_candidate_distinctness.py
- tests/test_dataset_learning_profile.py
- tests/test_description_encoding.py
- tests/test_extreme_ml_time_budget.py
- tests/test_fast_materializer.py
- tests/test_generation_performance_guards.py
- tests/test_gui_config_contract.py
- tests/test_gui_quality_messages.py
- tests/test_learning_palette_fallback.py
- tests/test_marker_spam_prevention.py
- tests/test_materializer_playability_preservation.py
- tests/test_mode_transition_planning.py
- tests/test_no_improvement_early_stopping.py
- tests/test_object_count_metadata_consistency.py
- tests/test_object_count_scaling.py
- tests/test_object_diversity.py
- tests/test_Gemini_json_recovery.py
- tests/test_playable_skeleton_generation.py
- tests/test_quality_draft_diagnostics.py
- tests/test_repair_loss_guard.py
- tests/test_report_text_sanity.py

## Behaviour changes already applied

| Area | Change | Verification |
|---|---|---|
| materializer | ~30-id safe decoration palette replaces `['211']` fallback | tests/test_learning_palette_fallback.py |
| materializer | Monotonic cursor walk + per-section sort + global stable sort | tests/test_object_count_scaling.py::test_high_object_count_remains_x_monotonic |
| materializer | Y-band separation (gameplay y<180, deco 180-380, bg 300-540) | tests/test_marker_spam_prevention.py |
| materializer | Role tagging (fill_decoration / background_detail) | tests/test_fast_materializer.py::test_decoration_avoids_gameplay_y_band |
| materializer | Per-section palette rotation across 4 motif families | tests/test_learning_palette_fallback.py::test_palette_rotation_avoids_uniform_section_output |
| audio_conditioned | Per-candidate mat_config seed offset (`+ candidate_id * 7919`) | tests/test_candidate_distinctness.py |
| learning/feature_extractor | `build_palette_from_learned_store` builds ids_by_class palette | tests/test_dataset_learning_profile.py |
| gui/app.py | Quality message `already_extreme` guard | tests/test_gui_quality_messages.py |
| gui/app.py | Diagnostic field display (playability, repair_loss, final_objects) | tests/test_gui_quality_messages.py |

## Measured improvements vs. previous failure baseline

| Metric | Previous failure | After port | Test |
|---|---|---|---|
| object_diversity_score | 0.0017 | ≥0.005 (synthetic 1k objs) | test_diversity_dramatically_above_0_0017_baseline |
| x_mono violations | many | 0 (3360 obj synthetic) | test_high_object_count_remains_x_monotonic |
| candidate distinctness | identical | ≥2 distinct sigs | test_different_seeds_produce_distinct_output |
| description encoding | base64 leak | single round-trip ok | test_description_in_export_is_base64_only_once |
| pytest | 540 (pre-redesign) | 607 | full suite |

## Assumptions

- The wip branch tree is the intended final state.
- pytest 607 / 17 skipped is the regression baseline.
- LICENSE is GPL-3.0; manifest SPDX is added in Phase J.

## Constraints

- No live Gemini call during pytest.
- No bulk object generation through AI; AI is structured-output only.
- No Gemini/OpenAI/Claude runtime providers.

## Risks

- Stale dependencies (`google-genai`, `openai`) declared in pyproject but not imported. Phase J removes them.
- A single test (`tests/test_dataset_index.py::test_default_dataset_dir_is_dataset`) asserts parent dir name == 'gmdgen' and is therefore brittle to clones at other paths.
- `gmdgen.gd.guidelines` and other modules not exhaustively diff-reviewed; rely on pytest as the regression net.
