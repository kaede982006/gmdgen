import os
import glob
from pathlib import Path
import re

for filepath in glob.glob("docs/**/*.md", recursive=True) + ["README.md", "CHANGELOG.md", "RELEASE_NOTES_0.1.0.md"]:
    p = Path(filepath)
    if not p.is_file(): continue
    text = p.read_text(encoding="utf-8")
    
    # Massive cleanup for Gemini-only CLI
    text = re.sub(r'(?i)ollama-only', 'Gemini-only', text)
    text = re.sub(r'(?i)ollama based', 'Gemini API based', text)
    text = re.sub(r'(?i)ollama/qwen', 'Gemini', text)
    text = re.sub(r'(?i)ollama planner', 'Gemini planner', text)
    text = re.sub(r'(?i)ollama', 'Gemini', text)
    text = re.sub(r'(?i)qwen2\.5-coder:7b', 'gemini-2.5-flash', text)
    text = re.sub(r'(?i)qwen2\.5-coder:3b', 'gemini-2.5-flash', text)
    text = re.sub(r'(?i)qwen3?:14b', 'gemini-2.5-flash', text)
    text = re.sub(r'(?i)qwen', 'Gemini', text)
    text = re.sub(r'localhost:11434', 'https://generativelanguage.googleapis.com', text)
    text = re.sub(r'OLLAMA_HOST', 'GEMINI_API_KEY', text)
    text = re.sub(r'does not require an API key', 'requires a GEMINI_API_KEY', text)
    text = re.sub(r'GUI-only for user generation', '', text)
    text = re.sub(r'GUI usage.*', '', text)
    text = re.sub(r'Generate button', '`gmdgen generate` command', text)
    text = re.sub(r'--provider openai', '--provider gemini', text)
    text = re.sub(r'--fallback-provider openai', '', text)
    text = re.sub(r'OpenAI fallback is available.*', 'No fallback providers are supported.', text)
    text = re.sub(r'Gemini and OpenAI are legacy.*', 'Gemini is the exclusive runtime provider.', text)
    text = re.sub(r'Gemini produces strict symbolic', 'Gemini produces strict symbolic', text)
    
    p.write_text(text, encoding="utf-8")

