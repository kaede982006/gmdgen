import sys
import re

content = open('src/gmdgen/gui/app.py', 'r', encoding='utf-8').read()

# Replace any occurrence of Ollama UI variables
content = re.sub(r'        self\.v_ollama_api_key = tk\.StringVar\(\)\n', '', content)
content = re.sub(r'        self\.v_ollama_model = tk\.StringVar\(value="gpt-4\.1-mini"\)\n', '', content)
content = re.sub(r'        self\.v_use_environment_key = tk\.BooleanVar\(value=True\)\n', '', content)
content = re.sub(r'        self\.v_fallback_to_ollama = tk\.BooleanVar\(value=False\)\n', '', content)
content = re.sub(r'        self\.v_ollama_strict_json = tk\.BooleanVar\(value=True\)\n', '', content)
content = re.sub(r'        self\.v_ollama_save_debug_artifacts = tk\.BooleanVar\(value=False\)\n', '', content)
content = re.sub(r'        self\.v_ollama_debug = tk\.BooleanVar\(value=False\)\n', '', content)

content = re.sub(r'        ollama_api_key: str = ""\n', '', content)
content = re.sub(r'        ollama_model: str = "gpt-4\.1-mini"\n', '', content)
content = re.sub(r'        use_environment_key: bool = True\n', '', content)
content = re.sub(r'        fallback_to_ollama: bool = False\n', '', content)

# Check to_generation_config calls
content = re.sub(r'                ollama_base_url=self\.v_ollama_api_key\.get\(\)\.strip\(\),\n', '', content)
content = re.sub(r'                use_environment_key=self\.v_use_environment_key\.get\(\),\n', '', content)
content = re.sub(r'                ollama_model=self\.v_ollama_model\.get\(\)\.strip\(\),\n', '', content)
content = re.sub(r'                fallback_to_ollama=self\.v_fallback_to_ollama\.get\(\),\n', '', content)
content = re.sub(r'                ollama_strict_json=self\.v_ollama_strict_json\.get\(\),\n', '', content)
content = re.sub(r'                ollama_save_debug_artifacts=self\.v_ollama_save_debug_artifacts\.get\(\),\n', '', content)
content = re.sub(r'                ollama_debug=self\.v_ollama_debug\.get\(\),\n', '', content)

# Check the UI setup
content = re.sub(r'            tab_ollama = ttk\.Frame\(notebook, padding=10\)\n            notebook\.add\(tab_ollama, text="Ollama / Legacy"\)\n            self\._row_secret\(tab_ollama, ttk, "Ollama API key \(legacy\)", self\.v_ollama_api_key\)\n            self\._row_check\(tab_ollama, ttk, "Use OPENAI_API_KEY env", self\.v_use_environment_key\)\n            self\._row_text\(tab_ollama, ttk, "Ollama model", self\.v_ollama_model\)\n            self\._row_check\(tab_ollama, ttk, "Fallback to Ollama if Ollama fails", self\.v_fallback_to_ollama\)\n', '', content)

# Remove run_ollama_only_audit imports and usage
content = re.sub(r'from gmdgen\.audit\.ollama_only import OllamaOnlyAuditResult, run_ollama_only_audit\n', '', content)
content = re.sub(r'    audit_result: OllamaOnlyAuditResult \| None = None\n', '    audit_result: Any | None = None\n', content)
content = re.sub(r'        self\.state\.audit_result = run_ollama_only_audit\(base_config, \{"provider_options": \["ollama", "ollama"\]\}\)\n', '', content)
content = re.sub(r'        audit = run_ollama_only_audit\(config_dict, \{"provider_options": \["ollama", "ollama"\]\}\)\n        if not audit\.passed:\n            raise RuntimeError\("External AI-only audit failed before generation"\)\n', '', content)
content = re.sub(r'    def run_ollama_only_audit\(self, config: dict\[str, Any\]\) -> OllamaOnlyAuditResult:\n        self\.state\.audit_result = run_ollama_only_audit\(config, \{"provider_options": \["ollama", "ollama"\]\}\)\n        return self\.state\.audit_result\n', '', content)
content = re.sub(r'            self\._row_button\(tab_ai, ttk, "Run Ollama-only Audit", self\._run_ollama_only_audit\)\n', '', content)

# Fix remaining mentions of ollama
content = content.replace('def _run_ollama_only_audit(self) -> None:', 'def _run_ollama_only_audit(self) -> None:')

# Worker logic fixing
content = re.sub(r'        should_set_env_key = bool\(self\.config\.ollama_api_key\.strip\(\)\)\n', '', content)
content = re.sub(r'        if should_set_env_key:\n            os\.environ\["OPENAI_API_KEY"\] = self\.config\.ollama_api_key\.strip\(\)\n', '', content)
content = re.sub(r'            if should_set_env_key:\n                if prior_key is None:\n                    os\.environ\.pop\("OPENAI_API_KEY", None\)\n                else:\n                    os\.environ\["OPENAI_API_KEY"\] = prior_key\n', '', content)
content = re.sub(r'        prior_key = os\.environ\.get\("OPENAI_API_KEY"\)\n', '', content)

# Replace remaining ollama
content = content.replace('ollama_package_available', 'ollama_package_available')

with open('src/gmdgen/gui/app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("done")
