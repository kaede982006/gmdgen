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
