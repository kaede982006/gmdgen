# gmdgen v0.1.0 — Clean GPL-3.0 baseline & algorithmic redesign

## Highlights

- **Clean GPL-3.0 baseline**: prior history archived locally, repository
  reset to a clean GPL-3.0-or-later baseline.
- **Algorithmic redesign applied**: Ollama is a strict symbolic planner, not a
  `.gmd` generator. Final output is accepted only through the local
  IR -> allocator -> serializer -> validator -> repair/report pipeline.
- **Legacy object-plan path sealed**: Ollama planner output containing raw
  `.gmd`, `object_plans`, `trigger_plans`, concrete group/color ids, scores, or
  validation verdicts is rejected before production generation.
- **Report consistency is a release gate**: planner fallback, candidate/final
  counts, validation state, low-quality drafts, and `final_success` are tracked
  separately so fallback drafts cannot be reported as finished levels.
- **Pattern fixture churn prevented**: pattern tests are read-only by default;
  fixture regeneration requires explicit maintenance opt-in.
- **Ollama-only runtime**: deterministic generation works without any AI;
  when Ollama is available, it produces structured section plans only.
- **252 source files** carry an `SPDX-License-Identifier: GPL-3.0-or-later`
  header.
- **690 tests pass**, 17 skipped (no live Ollama required).

## Performance

| Metric | Failure baseline | This release |
|---|---|---|
| object_diversity_score | 0.0017 | ≥0.005 (1k synthetic) |
| x_mono violations | thousands | 0 (3360 obj synthetic) |
| candidate distinctness | identical | ≥2 distinct signatures |
| pytest | 540 (pre-redesign) | 690 |

## Compatibility

- Python ≥ 3.10
- Optional runtime: Ollama with `qwen2.5-coder:7b` (or any compatible model)

## Dependencies

| Package | Constraint | License | GPL-3.0 compatible |
|---|---|---|---|
| PyYAML | ≥6.0 | MIT | ✅ |
| pytest (dev) | ≥8.0 | MIT | ✅ |

`google-genai` and `openai` were removed; they were declared but not
imported.

## License

GNU General Public License v3.0 or later. See `LICENSE`.

## Verification

```bash
python -m pip install .
python -m pytest -q
```

Expected: `690 passed, 17 skipped`.

Generated samples in `release_assets/` are deterministic and reproducible
given the same RawSpec + seed.
