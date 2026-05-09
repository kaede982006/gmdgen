# gmdgen v0.1.0 — AI baseline (Tier-2 LM)

## Highlights

- **Emergency Hotfix: Planner Stabilization (May 9)**: Fixed `ollama_forbidden_field` errors by hardening symbolic planner prompts, adding context filtering for forbidden keywords, and improving diagnostics/reporting consistency.
- **Real neural network**: the dataclass-only "AI" spec is replaced by
  ``GMDLanguageModel`` — a 4-layer causal Transformer (~600k parameters)
  that actually trains. forward / backward / optimizer step are exercised
  by ``python -m gmdgen.ml.smoke`` and a new ``tests/test_ml_smoke.py``.
- **Self-supervised pre-training on .gmd**: ``python -m gmdgen.ml.train``
  reads the ``dataset/`` corpus, slices each level into overlapping windows,
  and learns next-object cross-entropy with auxiliary ``dx`` / ``y`` heads.
  600-step CPU run reduces validation loss from 5.58 → 3.97 and held-out
  perplexity from 80 → 53.
- **Factorized tokenization**: each GD object is encoded as 7 sub-tokens
  (id / cls / dx / y / mode / speed / section). Each factor has its own
  learnable embedding; inputs are summed (Goodfellow Ch.15 — disentangling
  causal factors; Fleuret §4.9).
- **Constrained decoding**: nucleus (top-p=0.9) + top-k(40) + temperature
  sampling, plus a monotonic-x mask and a post-hoc ground-rail repair so
  every generated level satisfies the I-1..I-5 + R0 invariant suite.
- **CLI integration**: ``gmdgen generate --use-ml --ml-ckpt <path>``
  produces a real ``.gmd`` whose ``k4`` is the gzip+base64-encoded level
  string from sampler output.
- **Evaluation harness**: ``gmdgen.eval.metrics`` computes editor-load
  rate, play-success rate, mode-coverage KL, held-out perplexity,
  repair-loss proxy, and invariant-pass rate, and serialises them to
  ``reports/eval.json``.
- **Dataset audit**: ``tools/audit_dataset.py`` reports per-mode and
  per-class distributions so future training can debias the corpus
  (Onuoha; Smith & Rustagi).

## Verified metrics (seed=0, 8 samples, sections=4)

| Metric | Value |
|---|---|
| ``editor_load_rate`` | 1.00 |
| ``simulate_play_success_rate`` | 1.00 |
| ``mode_coverage_kl`` | 1.23 |
| ``repair_loss_proxy`` | 0.00 |
| ``held_out_perplexity`` | 62.36 |
| ``invariant_pass.rate`` | 1.00 |
| ``n_params`` | 608,022 |
| ``pytest`` | 702 passed, 17 skipped |

## PDF → code mapping (highlights)

| Reference | Implementation |
|---|---|
| Fleuret §3.2 / §3.3 | ``ml/train._shifted_ce`` + AdamW + cosine schedule |
| Fleuret §4.8–§4.10 | Causal-masked TransformerEncoder + sinusoidal positional encoding |
| Fleuret §5.3 / §5.7 | ``GMDLanguageModel`` GPT-style stack |
| Foundations of LLMs §1.1.1 | self-supervised next-token objective |
| Foundations of LLMs §5.1.3 | nucleus + top-k decoding in ``ml.sample`` |
| Boonstra | sampling defaults & few-shot Ollama planner prompts |
| Onuoha; Smith & Rustagi | dataset audit / debiasing report |

## Compatibility

- Python ≥ 3.10 (verified on 3.14).
- Optional ML stack: ``pip install gmdgen[ml]`` (PyTorch CPU + numpy).
- Optional runtime: Ollama with ``qwen2.5-coder:7b`` for the *symbolic*
  planner path (not the new ML path).

## Dependencies

| Package | Constraint | License | GPL-3.0 compatible |
|---|---|---|---|
| PyYAML | ≥6.0 | MIT | ✅ |
| torch | ≥2.2 (optional) | BSD-3 | ✅ |
| numpy | ≥1.26 (optional) | BSD-3 | ✅ |
| pytest (dev) | ≥8.0 | MIT | ✅ |

## Reproducing v0.1.0

```bash
pip install -e .[ml,dev]
python -m gmdgen.ml.smoke
python -m gmdgen.ml.train --in dataset --out ckpts/gmd_lm_tiny.pt \
    --max-steps 600 --batch 8 --ctx 256 \
    --log-jsonl reports/train_log.jsonl
python -m gmdgen generate --use-ml \
    --ml-ckpt ckpts/gmd_lm_tiny.pt --prompt "energetic neon ship" \
    --seed 7 --sections 4 --output outputs/ml_demo.gmd
python -m gmdgen.eval.metrics --ckpt ckpts/gmd_lm_tiny.pt \
    --in dataset --samples 8 --out reports/eval.json
python -m pytest -q
```

Expected: ``702 passed, 17 skipped``.

## Release assets

The trained checkpoint and generated metric reports are attached as release
assets when publishing. They are intentionally ignored by git so source
history does not absorb binary/runtime artifacts.

## License

GNU General Public License v3.0 or later. See ``LICENSE``.
