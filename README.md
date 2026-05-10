# gmdgen

Local Geometry Dash level generator with deterministic structure-first generation and Gemini API planning.

## Features

- Gemini API-first generation (`gemini-2.5-flash` by default).
- Deterministic and structured code generation workflow.
- Rich realtime logging with progress percentages.

## Usage

**Set the API Key:**
```bash
export GEMINI_API_KEY='your-key'
```

**Run Doctor Check:**
```bash
gmdgen doctor --check-provider-live
```

**Generate Level:**
```bash
gmdgen generate --audio song.wav --model gemini-2.5-flash
```

**Other Commands:**
- `gmdgen train`: Build dataset context
- `gmdgen validate <file>`: Validate a level
- `gmdgen repair <file>`: Repair a level
- `gmdgen report <file>`: Generate a report

## Migration & Notes

* **GUI**: The old GUI is deprecated and no longer the default path.
* **Ollama/Qwen**: Ollama and Qwen are no longer the default providers. Ollama runs locally and does not require an external API key (does not require an API key).
* **OpenAI Fallback**: OpenAI fallback is available only when explicitly requested via `--allow-fallback --fallback-provider openai`.
* **Output Structure**: All outputs are generated under `outputs/runs/`.

## v0.1.0 Release

* Restructured to a Gemini API-based CLI workflow.
* Removed Ollama and GUI dependencies from the critical path.
