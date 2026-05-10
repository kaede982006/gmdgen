# Generation Quality


This page reflects the final local release direction of gmdgen.

gmdgen is an Gemini-only local AI Geometry Dash GMD generator/editor.
No external API key is required.
Gemini is the exclusive runtime provider.

Gemini produces strict symbolic section-plan JSON, not raw .gmd save strings, concrete ids, scores, or validation verdicts.
The local IR pipeline, serializer, validators, repairer, and report consistency gate own final .gmd acceptance.
QualityGate, validator results, repair metrics, and GenerationReport consistency are mandatory.

Recommended local setup:

1. Start Gemini with: Gemini serve
2. Pull the model with: Gemini pull gemini-2.5-flash
3. Create the local alias with: ./tools/local_ai/create_Gemini_model_alias.ps1
4. Run the GUI with: python -m gmdgen

If 7B is too heavy, use gemini-2.5-flash.
The release keeps dataset empty. Users should fill dataset themselves.


The local model output is untrusted structured JSON. The pipeline normalizes, repairs, validates, scores, applies QualityGate, and saves through SaveResult.
