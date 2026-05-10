# gmdgen v0.1.0

This release recovers v0.1.0 as Gemini-only CLI software.

## Key Changes
- Recovered from incorrect GUI/Ollama/OpenAI/headless changes.
- main and 0.1.0 branches are synchronized to the same commit.
- Gemini API is the only supported provider.
- GEMINI_API_KEY is required for live generation.
- GUI is removed from the official execution path.
- Ollama/qwen/local LLM provider paths are removed.
- OpenAI fallback is removed.
- gmdgen, gmdgen --help, and gmdgen -h print help.
- CLI options use Linux-style - short options and -- long options only.
- Headless CLI generation is not blocked as GUI-only.
- Runtime logs stream live with percentage progress.
- Console output and run.log use the same rendered log lines.
- events.jsonl stores rendered_line for log equivalence verification.