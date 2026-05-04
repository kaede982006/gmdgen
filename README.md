# gmdgen

A local Geometry Dash level (`.gmd`) generator with deterministic
structure-first generation and optional Ollama-based AI planning.

The runtime AI provider is **Ollama only**. Gemini, OpenAI, and other
cloud providers are intentionally not supported as runtime providers.
Ollama is used to produce structured JSON plans (not raw `.gmd` save
strings); deterministic algorithms then materialize the level.

## Installation

```bash
python -m pip install --upgrade pip
python -m pip install .
```

Recommended Ollama model: `qwen2.5-coder:7b`.

## Running with Ollama

`gmdgen` uses [Ollama](https://ollama.com/) as its local inference backend.
All AI calls go through the Ollama HTTP API; no cloud provider is required.
Deterministic generation also works with Ollama unavailable — in that case
`gmdgen` falls back to its built-in safe palette and motif families.

### 1. Install Ollama

| Platform | Command |
|---|---|
| Linux    | `curl -fsSL https://ollama.com/install.sh \| sh` |
| macOS    | `brew install ollama` *or* download from [ollama.com](https://ollama.com) |
| Windows  | Download the installer from <https://ollama.com/download> |

Verify:

```bash
ollama --version
```

### 2. Pull the planner model

`gmdgen` uses one model for level planning (structured JSON output).

```bash
ollama pull qwen2.5-coder:7b
```

Alternatives:

- `qwen2.5-coder:3b` — smaller, lower memory; faster on CPU
- `qwen2.5:7b` — generic chat variant if `coder` is unavailable

### 3. Start the Ollama server

Most installs auto-start the server. If not:

```bash
# Linux (systemd, system-wide)
sudo systemctl enable --now ollama

# Linux (user, no sudo) — fallback foreground
ollama serve

# macOS
brew services start ollama

# Windows
# Ollama runs as a tray app after install; no command needed.
```

Recommended environment for `ollama serve` (set before starting):

| Variable | Suggested | Why |
|---|---|---|
| `OLLAMA_HOST` | `127.0.0.1:11434` | Default; change if you bind elsewhere |
| `OLLAMA_KEEP_ALIVE` | `30m` | Keeps model resident → no cold start |
| `OLLAMA_NUM_PARALLEL` | `2` | Matches gmdgen's typical concurrency |
| `OLLAMA_MAX_LOADED_MODELS` | `1` | Single planner model stays hot |

Quick sanity check:

```bash
curl -s http://127.0.0.1:11434/api/tags | python -m json.tool
```

### 4. Point `gmdgen` at your Ollama instance

`gmdgen` reads the following environment variables (verified against the code):

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` (Ollama default) | Ollama HTTP endpoint; the GUI also exposes a "Use OLLAMA_HOST env" checkbox |
| `GMDGEN_DATASET_DIR` | `dataset` | Reference levels directory used by the learning store |
| `GMDGEN_HEADLESS` | unset | Set to `1` to force headless mode (skip Tk init) |
| `RUN_OLLAMA_LIVE_TESTS` | unset | Set to `1` to enable optional live-Ollama test paths (off in CI) |

Model selection and other generation parameters (Ollama model name, candidate
count, object multiplier, quality mode, etc.) are configured through the GUI
or via `GuiGenerationConfig`, not through environment variables.

Example (POSIX):

```bash
export OLLAMA_HOST=http://127.0.0.1:11434
export GMDGEN_DATASET_DIR=/path/to/your/reference/levels
python -m gmdgen
```

Example (Windows PowerShell):

```powershell
$env:OLLAMA_HOST = "http://127.0.0.1:11434"
$env:GMDGEN_DATASET_DIR = "C:\path\to\references"
python -m gmdgen
```

### 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `connection refused :11434` | Ollama not running | Start the server (step 3) |
| First call takes 30+ s | Cold model load | Set `OLLAMA_KEEP_ALIVE=30m` |
| `model 'X' not found` | Model not pulled | `ollama pull X` |
| Output is free-form text, not JSON | Model lacks JSON-mode | Use `qwen2.5-coder:7b` |
| OOM on 8 GB RAM | Planner model too large | Use `qwen2.5-coder:3b` |
| Slow on CPU only | Expected | Use a 3B model or enable GPU per Ollama docs |

For deeper tuning (GPU layers, quantization, custom Modelfiles), see the
[Ollama documentation](https://github.com/ollama/ollama/blob/main/docs/).

## Usage

```bash
python -m gmdgen          # GUI
python -m pytest -q       # full test suite (no live Ollama required)
```

## Dataset

`dataset/` is intentionally empty in the released package; users provide
their own reference `.gmd` files. With an empty dataset, the generator
falls back to a built-in safe palette and motif families. With a populated
dataset, learned palettes and density profiles are used as priors.

## License

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version. See [LICENSE](LICENSE).

## Verification

```bash
python -m pytest -q
```

Expected: `607 passed, 17 skipped`.
