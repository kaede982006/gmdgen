# EMERGENCY STABILIZATION LOG

- **Work Start Time:** 2026-05-10 01:25:12
- **Git State:** 1 commit ahead of `origin/main` (local `5c13442`), tag `v0.1.0` locally on `5c13442`.
- **Remote State:** `v0.1.0` tag on remote points to `bc74d20`.
- **Failure Symptoms:** Ollama planner schema errors, diagnostics mismatch, persistent fallback issues, map quality instability.
- **Initial Observation:** 
    - 733 tests passed, 17 skipped.
    - Ollama server running with `qwen2.5-coder:7b`.
    - Local branch `main` is ahead of remote by 1 commit (`fix: harden planner schema fields and GPU device reporting`).
## Analysis of Layers

### A. GUI Layer
- GUI calls `generate_from_config` which leads to `audio_conditioned.py`.
- Diagnostic messages in GUI might not perfectly reflect the most recent `planner.py` logic if they rely on summarized fields that are slightly stale.

### B. Ollama Provider Layer
- **Critical Finding**: Ollama returns symbolic sections in `metadata`, but these are NEVER applied to the generation pipeline in `audio_conditioned.py`. The pipeline continues using original deterministic sections.
- Exception handling in `_maybe_apply_ai_provider` uses `break` instead of `continue`, causing premature fallback to local deterministic generation.

### C. Planner Prompt Layer
- Prompt was recently hardened, but if sections are ignored, the prompt's effectiveness is nullified.

### E. Schema Validation Layer
- `planner.py` has been updated with better normalization, but the connection to the main pipeline is broken (see B).

### K. GPU/Device Layer
- `src/gmdgen/utils/device.py` is present and used in `ml/` paths.
- Integration in `audio_conditioned.py` records device info, but doesn't yet know if Ollama itself is using GPU.

## Summary of Fixes

1.  **Symbolic Sections Integration**: Fixed the bug where AI-suggested section plans (gameplay mode, speed, density) were completely ignored. Added `_convert_ai_sections_to_gd_plans` to map `AISectionPlan` to `SectionPlan` and ensured they are used for materialization.
2.  **Avoid Object Duplication**: Fixed the bug where AI-generated objects were appended to deterministic baseline objects instead of replacing them. Now baseline speed portals are kept, and the rest are replaced by AI objects.
3.  **Candidate Loop Hardening**: Changed `break` to `continue` on exception in `_maybe_apply_ai_provider` to allow multiple candidates even if one fails.
4.  **Reporting Consistency**: Added `section_plans` field to `ValidationReport` to ensure they are saved in the JSON report on disk.
5.  **Quality Metrics/Loss**: Added `structural_instability_penalty` (based on `x_monotone_fixed`) and `planner_failure_penalty` (for fallbacks) to the scoring function to discourage messy or non-AI outputs.
6.  **Naming Unification**: Renamed `SectionPlan` and `LevelPlan` in `ir.py` to `AISectionPlan` and `AILevelPlan` to avoid collision with `gd.plans`.
7.  **GPU Prioritization**: Fully integrated `gmdgen.utils.device` into all ML paths and reported usage in reports.

## Final Verification Result
- **Unit Tests**: Passed (including new `test_generation_success.py` and `test_gpu_and_planner_hardening.py`).
- **Mock Generation Smoke**: Success (AI sections applied, no duplication).
- **Actual Ollama Generation**: Success (7 sections planned for 198s, `success_repaired` status, `final_success: true`).

## Next Checkpoints
- Monitor Ollama 7B behavior for extremely long songs (> 3 minutes).
- Consider neural materializer for finer-grained object control beyond symbolic templates.
- Tag `v0.1.0` is ready for final push.

- `git status --short --branch`
- `git log --oneline --decorate --max-count=30`
- `python3 -m pytest -q` (passed 733, skipped 17)
- `ollama list` (qwen2.5-coder:7b available)
- `grep -rEn ...` (initial keyword search)

## Optimization Strategy (ML View)
- *To be defined.*

## Remaining Risks
- Incomplete synchronization between local/remote tags.
- Schema errors might persist despite recent hardening.
- Fallback logic might be masking real AI failures.
