# Current GMD ML Notes From docs/AI

This note distills the PDFs in `docs/AI/` for the current `gmdgen` situation:
the tiny GMD language model trains and loads, but generated levels remain visually
and structurally weak, often collapsing toward straight-line object placement.

## Current Diagnosis

The original problem was not only model size. The sharper issue was the learning
contract between data representation, training objective, sampling, repair, and
evaluation. The May 9 update fixed the most direct contract violations and made
the remaining visual-quality gap measurable.

Observed project state:

- The model is a small causal Transformer over factorized object tokens.
- Training predicts `id`, `dx`, and `y`, while `cls`, `mode`, `speed`, and
  `section` are only conditioning streams.
- Sampling predicts only `id`, `dx`, and `y`; `cls`, `mode`, and `speed` are now
  reconstructed from sampled object ids with the same deterministic state
  machine used by training.
- Sampling enforces minimum monotonic spacing, prompt-primes game mode through
  portals when possible, samples multiple candidates, and applies coverage-aware
  support repair instead of a full flat rail.
- The dataset is dominated by decoration tokens.
- The object id vocabulary has a non-trivial unknown-token rate.
- Current metrics mostly test editor validity and coarse monotonic/playability
  properties, not artistic or gameplay structure.

## What The PDFs Imply

### 1. Inductive Bias Matters More Than Parameter Count First

Fleuret's discussion of underfitting, overfitting, and inductive bias maps
directly to this project. With limited curated level data, a raw next-object LM
has too little domain structure. The model needs bias that matches Geometry Dash:
section plans, speed-state timing, mode transitions, gameplay skeletons, motifs,
and separate decoration passes.

Immediate implication:

- Do not expect a larger Transformer alone to solve straight-line generation.
- Put more structure into the token stream and decoder before scaling.
- Treat the generator as structured sequence prediction, not raw object spam.

### 2. Train/Inference Mismatch Is A First-Order Bug

The LLM PDFs emphasize causal language modeling as a consistent contract:
training predicts the next token under the same conditioning pattern used during
generation. The previous sampler broke that contract by feeding synthetic factor
streams that the model did not learn as generated history. This is now fixed for
class, mode, and speed factors by reconstructing them from sampled ids.

Immediate implication:

- Keep deterministic factor reconstruction tested and visible in diagnostics.
- The next model iteration should either predict all factors autoregressively or
  formalize these hard state machines as part of the model contract.
- Section remains a coarse deterministic schedule; a learned planner is still
  needed for intentional long-range structure.

### 3. Perplexity Is Necessary But Not Sufficient

The NLP PDFs treat held-out likelihood and perplexity as intrinsic language-model
metrics, but also warn that intrinsic gains can fail to transfer to downstream
quality. For `gmdgen`, a lower id perplexity does not imply a better level.

Immediate implication:

- Keep perplexity for regression testing, but stop treating it as the quality
  headline.
- Add extrinsic metrics that measure generated levels directly.
- Split validation by level, not by overlapping windows, to avoid optimistic
  validation numbers.

### 4. Vocabulary And Sparsity Need Better Handling

The NLP material on out-of-vocabulary handling and smoothing applies to GMD
object ids. Collapsing rare objects to `UNK` destroys useful style and trigger
information. At the same time, modeling every object id at equal granularity is
data hungry.

Immediate implication:

- Increase or adapt the object-id vocabulary after measuring rare-but-important
  ids such as portals, triggers, or gameplay objects.
- Consider hierarchical ids: semantic role first, concrete object id second.
- Downsample or separate decoration so frequent decorative ids do not dominate
  gameplay structure.

### 5. Long Context Needs Hierarchy

The long-sequence sections make the tradeoff clear: longer attention is expensive
and not automatically enough. GMD levels have long-range structure, but the
practical path is hierarchical modeling.

Immediate implication:

- Generate section-level plans before object-level tokens.
- Train on section/motif chunks with explicit boundary tokens.
- Use reference motif retrieval or candidate ranking for local detail instead of
  asking one tiny LM to invent a full level end to end.

### 6. Candidate Generation Plus Verification Fits This Domain

The prompting and RLHF sections are useful mostly as a design analogy:
generate multiple candidates, score them with a verifier/reward model, then
select or refine. Geometry Dash has many constraints that are easier to verify
than to learn from raw cross-entropy.

Immediate implication:

- Sample multiple continuations per section.
- Score candidates with playability, density, motif diversity, rail-collapse,
  mode-transition, and trigger-validity metrics.
- Keep repair as a measurable post-process, not as a hidden transformation that
  makes bad samples look valid.

## Immediate Engineering Priorities

1. Keep reducing repair dependency; high repair ratios still indicate the LM is
   not independently producing enough playable support.
2. Rebalance the dataset toward gameplay-critical objects.
3. Move toward a two-stage generator:
   - section/gameplay skeleton
   - object/materialization/deco pass
4. Increase or adapt the object-id vocabulary for rare but important gameplay
   ids.
5. Train longer only after the structural metrics improve under the same
   evaluation harness.

Completed in the May 9 update:

1. Fixed the sampler/training factor mismatch for class, mode, and speed.
2. Added level-wise train/validation splitting.
3. Added structural metrics:
   - rail or straight-line dominance
   - decoration-to-gameplay ratio
   - repeated-object entropy
   - section density variance
   - mode/speed transition coverage
   - reference motif similarity
4. Added prompt-primed candidate sampling and exposed repair ratio in eval.

## Practical Next Experiment

The next useful experiment is no longer another blind 600-step run. A better
experiment is:

1. Build a gameplay-skeleton dataset where support, hazards, portals, and speed
   changes are modeled before decoration.
2. Train a small skeleton model and a separate decoration/materialization pass.
3. Compare against the current one-pass model using repair ratio, rail
   dominance, straight-line dominance, mode coverage, and manual visual review.
4. Only then scale model size, context length, and training steps.

This follows the main lesson from the PDFs: model performance is shaped by the
objective, representation, data distribution, and inference procedure together.
