# Low Cost Mode


This page reflects the final local release direction of gmdgen.

gmdgen is an Ollama-only local AI Geometry Dash GMD generator/editor.
No external API key is required.
Gemini and OpenAI are legacy, retired, non-runtime providers in this release.

Ollama produces strict symbolic section-plan JSON, not raw .gmd save strings, concrete ids, scores, or validation verdicts.
The local IR pipeline, serializer, validators, repairer, and report consistency gate own final .gmd acceptance.
QualityGate, validator results, repair metrics, and GenerationReport consistency are mandatory.

Recommended local setup:

1. Start Ollama with: ollama serve
2. Pull the model with: ollama pull qwen2.5-coder:7b
3. Create the local alias with: ./tools/local_ai/create_ollama_model_alias.ps1
4. Run the GUI with: python -m gmdgen

If 7B is too heavy, use qwen2.5-coder:3b.
The release keeps dataset empty. Users should fill dataset themselves.


Low Cost mode limits local compute, candidates, prompt size, and retry count. It does not reduce external API spending because this release does not require external API keys.
