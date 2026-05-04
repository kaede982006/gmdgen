# gmdgen v0.1.0 — Clean GPL-3.0 baseline & algorithmic redesign

## Highlights

- **Clean GPL-3.0 baseline**: prior history archived locally, repository
  reset to a clean GPL-3.0-or-later baseline.
- **Algorithmic redesign already applied**: x-monotone-by-construction
  generation, 30+-ID diverse safe palette (replacing degenerate `["211"]`
  fallback), role-tagged objects, per-section palette rotation, per-candidate
  materialization seed variation.
- **Ollama-only runtime**: deterministic generation works without any AI;
  when Ollama is available, it produces structured JSON plans only —
  bulk object generation stays deterministic.
- **252 source files** carry an `SPDX-License-Identifier: GPL-3.0-or-later`
  header.
- **607 tests pass**, 17 skipped (no live Ollama required).

## Performance

| Metric | Failure baseline | This release |
|---|---|---|
| object_diversity_score | 0.0017 | ≥0.005 (1k synthetic) |
| x_mono violations | thousands | 0 (3360 obj synthetic) |
| candidate distinctness | identical | ≥2 distinct signatures |
| pytest | 540 (pre-redesign) | 607 |

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

Expected: `607 passed, 17 skipped`.

Generated samples in `release_assets/` are deterministic and reproducible
given the same RawSpec + seed.
