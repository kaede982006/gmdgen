# Ollama Provider

Ollama is a strict planner provider. Its accepted output is the symbolic
`level_plan` + `sections` contract described in
`docs/ollama_planner_architecture.md`. The provider rejects raw `.gmd`, legacy
direct `object_plans`, concrete group/color ids, scores, and validation
verdicts.

Ollama Error Codes:
- ollama_server_unavailable
- ollama_model_missing
- ollama_timeout
- ollama_network_error
- ollama_invalid_json
- ollama_invalid_schema
- ollama_raw_gmd_rejected
- ollama_empty_response
- ollama_unknown_error
