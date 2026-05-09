# Emergency Stabilization Log

**Start Time**: 2026-05-09
**Current Git Status**: Cleaned up and stabilized. All fixes committed and pushed.
**Current Remote Status**: `origin` is up to date with `main` and `v0.1.0`.
**Tests**: 710 passed, 17 skipped. (Verified all tests pass).
**Current Failure Symptoms**:
Resolved `ollama_forbidden_field` by:
1. Strengthening the system prompt in `ollama_provider.py`.
2. Filtering the context provided to Ollama to remove forbidden keywords.
3. Updating report consistency rules in `report_consistency.py`.
4. Fixing the repair pass prompt.

## Checklist
- [x] Establish Log
- [x] Analyze Debug Log and JSON Report
- [x] Investigate `ollama_forbidden_field`
- [x] Fix Prompt/Context in `planner.py` or related
- [x] Add Repair Pass
- [x] Fix Report Consistency
- [x] Verify (Integration test with qwen2.5-coder:7b succeeded)
- [x] Commit & Push & Release

## Command Log
- `git status`, `git remote`, etc.
- `pytest` run (Fixed failure in `test_code_validation.py`).
- `python temp_test_198.py` (Integration test with Ollama: SUCCESS).
- `git push origin main` (SUCCESS).
- `git tag -f v0.1.0 && git push origin v0.1.0 -f` (SUCCESS).
- `gh release upload v0.1.0 dist/* --clobber` (SUCCESS).

## Final State
- **Git Commit**: 8aec5111717b6dd9c2f462e9466904eae0a51683
- **GitHub Release**: v0.1.0 updated with hotfix.
- **Diagnostics**: `raw_ollama_response_preview` and `extracted_json_preview` are now reliably captured and verified by consistency tests.
- **Reliability**: Headless generation with `qwen2.5-coder:7b` confirmed the fix for the 198-second Stereo Madness style prompt.

---

## 2026-05-09 Schema-Shape Stabilization (Current Pass)

### Immediate checks executed
- `git status --short --branch`: `main...origin/main`, existing local change in this log file.
- `git remote -v`: origin = `git@github.com:kaede982006/gmdgen.git`
- `git log --oneline --decorate --max-count=20`: HEAD at `8aec511`.
- `python -m compileall -q src tests scripts tools`: pass.
- `python -m pytest -q`: 5 failures unrelated to planner schema changes (`/home/xisik/.cache/gmdgen/logs` write denied in sandboxed environment).
- `rg` audit across `src/tests/docs/README`: planner diagnostics and fallback paths identified in `ai/planner.py`, `ai/ollama_provider.py`, `generate/audio_conditioned.py`, `validation/report_consistency.py`, `gui/app.py`.

### Root cause observed
- Current bad payload shape seen in field reports: `{"level_plan":{"sections":[]}}`
- Failure type is schema shape / required field mismatch, not forbidden-key violation.
- Missing pieces in prior diagnostics:
  - no explicit `missing_required_fields` / `wrong_location_fields`
  - no repair-pass attempt marker for repairable shape errors.
