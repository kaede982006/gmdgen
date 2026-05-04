# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic
Versioning.

## [0.1.0] - 2026-05-05

Clean GPL-3.0 baseline with the algorithmic redesign already applied.

### Added
- Diverse safe decoration palette (≥30 unique IDs, replaces the previous
  degenerate `["211"]` fallback that collapsed object_diversity_score).
- Per-section motif family rotation across 4 families.
- `build_palette_from_learned_store()` derives `ids_by_class` palette from a
  populated `LearnedDataStore`; falls back to the safe palette when empty.
- Per-candidate materialization seed offset (`+ candidate_id * 7919`) so
  candidates are distinct even under deterministic fallback.
- 33+ new tests covering: playable skeleton, materializer playability,
  repair-loss guard, object diversity, mode transition, quality diagnostics,
  GUI quality messages, performance guards, dataset learning profile,
  learning palette fallback, report text sanity, candidate distinctness,
  description encoding, marker-spam prevention, object count metadata,
  fast materializer extensions, object count scaling.
- SPDX-License-Identifier headers on all 252 Python source files.

### Changed
- Materializer fill decorations now walk a monotonic x cursor and stable
  sort within each section; `materialize_level_plans` finishes with one
  global stable sort. Result: 0 x_mono violations on 3360-object synthetic
  levels (previous baseline produced thousands of violations).
- Y-band separation: gameplay corridor `y < 180` reserved; decorations sit
  in `[180, 380]`, background details in `[300, 540]`.
- GUI quality warning now uses `already_extreme` guard before suggesting
  Extreme ML and includes Playability / Repair loss / Final objects /
  Stopped reason as readable diagnostics.
- pyproject license metadata: `license = { text = "GPL-3.0-or-later" }` and
  classifier added.

### Removed
- Stale dependencies `google-genai` and `openai` removed from pyproject.
  These were declared but not imported (verified via `grep -rn "import"`).
  Defensive sanitization of `gemini_` / `openai_` config keys remains in
  `factory.py` and `store.py`.

### Fixed
- Description encoding round-trip: single base64 encode for the GD `k3` tag,
  no double-encoding, UTF-8/Korean safe.
- Candidate identical-output regression: deterministic expansion now varies
  by `candidate_id`.

### Security
- No credentials are stored or transmitted.
- `git log` shows a single contributor (`kaede982006`); GPL relicensing is
  trivially compliant.

[0.1.0]: https://github.com/kaede982006/gmdgen/releases/tag/v0.1.0
