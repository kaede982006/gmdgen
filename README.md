# gmdgen

gmdgen is an Ollama-only local AI Geometry Dash GMD generator/editor.
No external API key is required.
Gemini and OpenAI are not runtime providers.
Ollama generates structured JSON plans, not raw .gmd save strings.

## Installation & Usage
Recommended model: qwen2.5-coder:7b (Optional alias: gmdgen-coder-final)
Run GUI: `python -m gmdgen`

## Dataset
dataset/ is intentionally empty in the release; users fill dataset/ themselves.

## Verification
Run tests: `python -m pytest -q` or `tools/local_ai/verify_final_release.ps1`
