# AI Baseline — v0.1.0 (Tier-2 Language Model)

This release converts the AI subsystem from a *spec-only* artifact into a
real, trained **Geometry Dash level language model**. The architectural
descriptions in `src/gmdgen/ml/architectures.py` are preserved for the
long-term audio-conditioned plan, but the actual generation now goes
through a small causal Transformer trained on the project's `.gmd` corpus.

## What changed

| Layer | Before | After |
|---|---|---|
| Tokenizer | string-join `OBJ:id\|CLS:S\|...` | factorized integer fields (`id`, `cls`, `dx`, `y`, `mode`, `speed`, `section`) |
| Sequence model | order-2 Markov counter | 4-layer causal Transformer (~600k params) |
| Training signal | frequency counts | next-object cross-entropy + auxiliary `dx` / `y` heads |
| Optimizer | n/a | AdamW(lr=3e-4, β=(0.9,0.95), wd=0.1) + cosine + 100-step warmup |
| Decoding | pattern lookup | nucleus (top-p=0.9) + top-k(40) + temperature(0.8), monotonic-x mask, ground-rail repair |
| Evaluation | none | held-out perplexity, editor-load rate, simulate-play success, mode-coverage KL, repair-loss proxy, invariant-pass rate |

## Theoretical mapping (PDF → code)

| PDF / chapter | Implemented as |
|---|---|
| Fleuret §3.2 (autoregressive) | `gmdgen.ml.train._shifted_ce` next-token CE |
| Fleuret §3.3 (gradient descent) | `torch.optim.AdamW` + cosine schedule + grad clip |
| Fleuret §4.8 (attention) | `nn.TransformerEncoderLayer` causal-masked, `norm_first=True` |
| Fleuret §4.9 (token embedding) | 7 separate `nn.Embedding` tables summed into d_model |
| Fleuret §4.10 (positional encoding) | `_SinusoidalPositionalEncoding` |
| Fleuret §5.3 / §5.7 (Transformer / GPT) | `GMDLanguageModel` decoder-only stack |
| Foundations of LLMs §1.1.1 (self-supervised) | `GMDTokenDataset` with shifted target |
| Foundations of LLMs §2.3 (long sequence) | sliding window with stride=ctx//2 |
| Foundations of LLMs §5.1.3 (decoding) | `_filter_logits` nucleus + top-k |
| Boonstra (LLM config) | sampling defaults `t=0.8, top_p=0.9, top_k=40` |
| Onuoha + Smith&Rustagi (bias) | `tools/audit_dataset.py` per-mode/class report |

## Reproducing v0.1.0

```bash
# Optional ML extras
pip install gmdgen[ml]   # or: pip install torch numpy

# 1) Audit the dataset (informational)
python tools/audit_dataset.py --in dataset --out reports/dataset_audit.json

# 2) Train the tiny LM (~30s on CPU, 600 steps)
python -m gmdgen.ml.train \
    --in dataset --out ckpts/gmd_lm_tiny.pt \
    --max-steps 600 --batch 8 --ctx 256 \
    --log-jsonl reports/train_log.jsonl

# 3) Smoke check (no checkpoint required)
python -m gmdgen.ml.smoke

# 4) Generate one .gmd from the trained checkpoint
python -m gmdgen generate --use-ml \
    --ml-ckpt ckpts/gmd_lm_tiny.pt \
    --prompt "energetic neon ship at 140bpm" \
    --seed 7 --sections 4 \
    --output outputs/ml_demo.gmd

# 5) Evaluate
python -m gmdgen.eval.metrics \
    --ckpt ckpts/gmd_lm_tiny.pt --in dataset \
    --samples 8 --out reports/eval.json
```

## Verified metrics (seed=0, 8 samples, sections=4)

```
editor_load_rate            : 1.00
simulate_play_success_rate  : 1.00
mode_coverage_kl            : 1.23
repair_loss_proxy           : 0.00
held_out_perplexity         : 62.36
invariant_pass.rate         : 1.00
n_params                    : 608,022
completed_steps             : 600
final_val_perplexity        : 53.19
pytest                      : 702 passed, 17 skipped
```

## Known limitations / next steps

* **No audio conditioning yet.** `librosa` is optional; `encode_audio` is a
  zero-vector stub. Adding a CNN-Transformer audio encoder + cross-attention
  is the natural next release.
* **No RLHF.** The reward functions in `eval/metrics.py` are computed but not
  fed back into training. PPO + a reward model is the v0.2.0 plan.
* **Tiny model.** 600k parameters is enough to learn local rhythm/structure
  from 86 levels but it cannot capture long-range trigger-graph semantics.
  Scaling to ~10M parameters with better data balance is the next priority.
* **Mode bias.** `tools/audit_dataset.py` shows `spider`/`robot` are
  underrepresented (19 / 33 occurrences). Augmentation weights should be
  tuned to compensate.
