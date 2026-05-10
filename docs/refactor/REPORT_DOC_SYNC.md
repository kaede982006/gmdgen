# Doc-Code Sync Report

| Axis | Status | Notes |
|---|---|---|
| D-1 CLI interface | PASS | flags in code but not in README: ['--ai-provider', '--audio-file', '--config', '--dataset', '--export-finetune-jsonl', '--low-cost-mode', '--max-ai-calls', '--min-val', '--no-Gemini-fallback', '--Gemini-context-dir'] |
| D-2 env vars | PASS | env vars in code: ['GMDGEN_CACHE_DIR', 'GMDGEN_DATASET_DIR', 'GMDGEN_HEADLESS', 'GMDGEN_LOG_LEVEL', 'GMDGEN_NO_PROGRESS', 'Gemini_HOST', 'RUN_Gemini_LIVE_TESTS']; env vars in code ⊆ README |
| D-3 dependencies | PASS | runtime deps in pyproject: ['PyYAML'] |
| D-4 markdown blocks | PASS | all extracted code blocks parse |
| D-5 changelog vs git | PASS | git: 0 commits beyond main; [0.1.0] section present |
| D-6 architecture | PASS | no docs/ARCHITECTURE.md (skipped) |
| D-7 markdown links | PASS | all internal markdown links resolve |
| D-8 SPDX coverage | PASS | all source files carry SPDX header |

**Overall: PASS**
