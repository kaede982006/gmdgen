# Logging & Observability Report — v2.3 Patches

## Module map

| Module | Purpose |
|---|---|
| `gmdgen.observability.log` | Structured JSONL logger + `@logged` decorator |
| `gmdgen.observability.progress` | Lightweight stderr progress bar |

## Schema

Every JSONL line carries:

| Field | Type | Source |
|---|---|---|
| `ts` | ISO-8601 string | `datetime.now(timezone.utc)` |
| `run_id` | `<UTC-ts>-<8 hex>` | first call to `get_logger()` |
| `phase` | string | call site |
| `step` | string | call site |
| `event` | string | one of `phase_start / phase_end / phase_error / cache_hit / ...` |
| `payload` | dict | optional |
| `level` | int | 0..3 (QUIET/NORMAL/VERBOSE/TRACE) |
| `elapsed_ms` | int | since logger start |
| `mem_mb` | float | best-effort (POSIX `resource` or `psutil`) |

## Phase coverage

The decorator surface covers (or can be applied to) the canonical 11-phase
pipeline. Currently wired:

| Phase | Step | Wired |
|---|---|---|
| `layout` | `expand_plan` | ✅ |
| `pipeline` | n/a | available via `session()` |
| `plan` / `decorate` / `validate` / `encode` / `ai` / `learning` / `cache` | n/a | available via `@logged(phase=...)` |

The infrastructure is ready; legacy modules are not yet decorated to avoid
risking the 607-test invariant. The `@logged` decorator is opt-in.

## Test coverage

`tests/test_observability_log.py`:

- JSONL writes succeed and are parseable.
- `@logged` records `phase_start` and `phase_end` with `duration_ms`.
- Exceptions are recorded as `phase_error` and re-raised.
- `LogLevel.QUIET` suppresses NORMAL events (filtering works).
- `run_id` is unique across sessions.

## Sample output (NORMAL level, stderr)

```
[layout] expand_plan phase_start
[layout] expand_plan phase_end duration_ms=2
```

JSONL row example:

```json
{"ts": "2026-05-05T05:30:00+00:00", "run_id": "20260505-053000-deadbeef",
 "phase": "layout", "step": "expand_plan", "event": "phase_end",
 "payload": {"duration_ms": 2}, "level": 1, "elapsed_ms": 2, "mem_mb": 67.3}
```

## Progress bar

`ProgressBar(total, desc, unit)` is a context-manager-ready stderr bar that
auto-silences when:

- `GMDGEN_LOG_LEVEL=0` (QUIET)
- `GMDGEN_NO_PROGRESS=1`
- stderr is not a TTY (CI safety)

Sample:

```
  layout      66.7% [################--------] 16/24 it (8/s)
```

## Environment variables

| Var | Purpose | Default |
|---|---|---|
| `GMDGEN_LOG_LEVEL` | 0..3 verbosity | 1 |
| `GMDGEN_NO_PROGRESS` | disable progress bars | unset |
| `GMDGEN_CACHE_DIR` | base for `~/.cache/gmdgen` | XDG cache home |

## Constraints

- Logging never raises; telemetry must not break generation.
- No external dependency (no `tqdm`, no `psutil` required — both used
  if present, no-op fallback otherwise).
