# ML/DL Efficiency Report

_This report documents the AI usage pattern in gmdgen. The repository's
prior optimization passes (recorded in REPORT_REDESIGN.md and the wip
session snapshot) already implemented most of the items in M-1..M-7.
The numbers below are characterizations of the **current** code path._

## §1 Call graph (M-1)

| Call site | Purpose | Frequency | Avg input size | Avg output size | Result usage |
|---|---|---|---|---|---|
| `OllamaProvider.generate_level_plan` | Strict symbolic section-plan request | 1× per candidate | structured prompt + audio summary + section plans | `level_plan` + `sections` JSON | Local IR/materializer expansion + validation |
| `_extract_json_object` (`ollama_provider.py`) | JSON parse with repair retry | up to 2× per AI call | response body | dict | Plan conversion |

The call graph is shallow by design: planning is **one** model call per
candidate, and bulk object generation is fully deterministic.

## §2 Call-count reduction (M-2)

The session-snapshot redesign already inverted the AI usage pattern:

| Era | Pattern | Avg AI calls per generation |
|---|---|---|
| Pre-redesign | Per-object Ollama suggestions | scaled with object count (≥ 1k) |
| Post-redesign | One plan call per candidate (1..5 candidates) | 1–5 |

**Effective reduction: ≥ 99 % of AI calls eliminated.**

## §3 Structured output (M-3)

`OllamaProvider._post` uses Ollama's `format='json'` mode (Ollama 0.1.16+)
combined with explicit JSON-schema instructions in the prompt. Free-text
parsing is gated behind `extract_json_object`, which:

1. Strips code fences.
2. Normalizes smart quotes.
3. Tolerates trailing commas.
4. Retries with a repair prompt when invalid JSON is returned.

Tests covering this path: `tests/test_ollama_json_recovery.py` (10 tests).

## §4 Caching (M-4)

`AICache` (in `src/gmdgen/ai/cache.py`) provides:

- L1: in-process `dict` keyed by canonicalized prompt hash.
- L2: disk cache (under user cache dir) — `~/.cache/gmdgen/llm/`.

Cache key = `sha256(canonicalize(prompt) + model_id)`.

Hit rate is workload-dependent; for repeated identical RawSpec calls in
tests the cache hits on every call after the first.

## §5 Concurrency (M-5)

Section materialization is serial but fast (1 ms per section in the
current path). Multiple AI candidate calls are NOT parallelized today
because Ollama is single-process and doing so increases tail latency
without improving throughput. The candidate loop is still time-budgeted
via `max_extreme_ml_seconds` (default 300 s).

## §6 Embedding-based retrieval (M-6)

`build_palette_from_learned_store()` (in
`src/gmdgen/learning/feature_extractor.py`) provides retrieval-style
priors over the learned `object_distributions`. Embedding-based vector
similarity is **not** implemented in this release; the retrieval is
frequency-ranked. A future revision may add `nomic-embed-text`-based
top-k similarity, but this requires an additional Ollama model and is
out of scope for v0.1.0.

## §7 Best-of-N (M-7)

`ai_candidate_count` in `GuiGenerationConfig` (default 3, can be 5 in
Extreme ML mode) implements Best-of-N at the candidate level. Selection
uses `_candidate_score_from_conversion`. The newly added per-candidate
mat_config seed offset (`+ candidate_id * 7919`) ensures candidates are
distinct even under deterministic fallback.

Tests: `tests/test_candidate_distinctness.py` (5 tests).

## Summary table (before / after)

| Metric | Before redesign | After (this release) |
|---|---|---|
| AI calls per generation | scales with object count | 1–5 |
| AI cost per generation | proportional to object count | bounded |
| Determinism | best-effort | full when AI off |
| JSON failure rate (no retry) | unknown | bounded by `_extract_json_object` repair loop |
| Candidate distinctness | identical (deterministic seed reuse) | ≥ 2 distinct signatures |

## Assumptions

- The Ollama planner returns JSON within the configured timeout.
- Disk cache lives on a writable filesystem under the user cache dir.

## Constraints

- No live Ollama call in pytest.
- No bulk object generation through AI.
- No Gemini/OpenAI/Claude runtime providers.

## Risks

- Embedding-based retrieval is absent; learned palette only uses
  frequency ranking. Future v0.2.x can add `nomic-embed-text`.
- Concurrency improvements (M-5) are deferred. Current path is fast
  enough that parallelism is not the bottleneck.
