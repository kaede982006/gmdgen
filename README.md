# gmdgen

gmdgen is a Gemini API-exclusive, headless CLI software for generating Geometry Dash levels using audio conditioning and AI planning.

## Features & Architecture

* **Gemini API Exclusive**: This project has been fully converted to a Gemini API-only CLI software. The primary provider is Gemini, and legacy paths (Ollama, local LLMs, OpenAI, and their respective fallbacks) have been completely removed.
* **API Key Required**: You must provide a `GEMINI_API_KEY` to run live generation.
* **Model Selection**: You can specify the model via the `GEMINI_MODEL` environment variable or the `--model` CLI option. The recommended and default model is `gemini-2.5-flash`.
* **Headless Operations**: All generations run headlessly. No GUI blocks or `GUI-only` requirements remain in the execution pathways.
* **Strict POSIX Options**: The CLI adheres exclusively to Linux/POSIX-style short (`-h`, `-q`) and long (`--help`, `--provider`) options.

## Getting Started

gmdgen, `gmdgen --help`, and `gmdgen -h` all print the main CLI help menu.

### Verifying your Environment

Use `gmdgen doctor` to check your environment, dependencies, and API key.
```bash
# Verify your GEMINI_API_KEY and perform a live API smoke test
export GEMINI_API_KEY='your-key-here'
gmdgen doctor --check-provider-live
```

### Generating a Level

Use `gmdgen generate` to run the main pipeline. 
```bash
gmdgen generate --audio-file my_song.wav --model gemini-2.5-flash
```

## Logging & Quality Gates

gmdgen utilizes a strictly unified logging and verification pipeline to ensure maximum observability and precision.
* **Real-time Logging & Progress**: Console logs stream live in real-time, displaying category, stage, and strict percentage-based progress indicators.
* **Unified Console and run.log**: The human-readable lines emitted to your console are identical, character-for-character, to the lines written to `outputs/runs/<id>/run.log`.
* **events.jsonl Integrity**: The `outputs/runs/<id>/events.jsonl` file records structured JSON logs. Each event contains a `rendered_line` attribute that is guaranteed to match the exact output found in the console and `run.log`.
* **Quality Gate Failure Policy**: If the generated level does not pass validation or contains zero objects, it will **not** be saved as `generated.gmd`. Low quality drafts will only be saved as `low_quality_draft.gmd` if you explicitly provide the `--allow-low-quality-draft` flag.

## Release & Branch Strategy

The `main` branch and `0.1.0` branch operate on the same release line and point to the same commit.

### v0.1.0 Release Changes
- Recovered from incorrect GUI/Ollama/OpenAI/headless legacy modifications.
- `main` and `0.1.0` branches are fully synchronized.
- Gemini API is now the sole supported provider.
- Headless CLI generation works correctly without GUI-only blockage.
- Unified, synchronous logging across console, `run.log`, and `events.jsonl`.
