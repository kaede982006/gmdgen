# SPDX-License-Identifier: GPL-3.0-or-later
import sys
import os
import re

def process_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove environment API key stuff
    content = content.replace('def environment_api_key_available(env_name: str = "OPENAI_API_KEY") -> bool:\n    return bool(os.environ.get(env_name, "").strip())\n', 'def environment_api_key_available(env_name: str = "GEMINI_API_KEY") -> bool:\n    return bool(os.environ.get(env_name, "").strip())\n')
    content = content.replace('def ollama_package_available() -> bool:\n    return importlib.util.find_spec("ollama") is not None\n\n', '')
    
    # redact_text
    content = content.replace('env_key = os.environ.get("OPENAI_API_KEY", "")\n', '')
    content = content.replace('    if env_key:\n        result = result.replace(env_key, "[REDACTED_OPENAI_API_KEY]")\n', '')
    
    # summarize generation error
    content = content.replace('Ollama schema error', 'Ollama schema error')
    content = content.replace('Ollama structured output schema is invalid', 'Ollama structured output schema is invalid')

    # worker
    content = content.replace('prior_key = os.environ.get("OPENAI_API_KEY")\n', '')
    content = content.replace('        if should_set_env_key:\n            os.environ["OPENAI_API_KEY"] = self.config.ollama_api_key.strip()\n', '')
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

process_file('src/gmdgen/gui/app.py')
print("Done")
