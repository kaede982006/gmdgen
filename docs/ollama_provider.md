# Gemini Provider

Gemini is a strict planner provider. Its accepted output is the symbolic
`level_plan` + `sections` contract described in
`docs/Gemini_planner_architecture.md`. The provider rejects raw `.gmd`, legacy
direct `object_plans`, concrete group/color ids, scores, and validation
verdicts.

Gemini Error Codes:
- Gemini_server_unavailable
- Gemini_model_missing
- Gemini_timeout
- Gemini_network_error
- Gemini_invalid_json
- Gemini_invalid_schema
- Gemini_raw_gmd_rejected
- Gemini_empty_response
- Gemini_unknown_error
