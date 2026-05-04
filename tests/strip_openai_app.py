import sys
import re

content = open('src/gmdgen/gui/app.py', 'r', encoding='utf-8').read()

# Remove Ollama notices
content = re.sub(r'OPENAI_REQUIRED_LABEL = AI_PROVIDER_REQUIRED_LABEL\nOPENAI_GENERATOR_NOTICE = AI_GENERATOR_NOTICE\nOPENAI_FAILURE_NOTICE = AI_FAILURE_NOTICE\n\n', '', content)

# Remove attributes from config
content = re.sub(r'        ollama_model: str = ".*?"\n', '', content)
content = re.sub(r'        ollama_api_key: str = ""\n', '', content)
content = re.sub(r'        use_environment_key: bool = True\n', '', content)
content = re.sub(r'        fallback_to_ollama: bool = False\n', '', content)
content = re.sub(r'        ollama_strict_json: bool = True\n', '', content)
content = re.sub(r'        ollama_save_debug_artifacts: bool = False\n', '', content)
content = re.sub(r'        ollama_debug: bool = False\n', '', content)

# Remove their uses in to_generation_config
content = re.sub(r'            "ollama_strict_json": self\.ollama_strict_json,\n', '', content)
content = re.sub(r'            "ollama_save_debug_artifacts": self\.ollama_save_debug_artifacts,\n', '', content)
content = re.sub(r'            "ollama_debug": self\.ollama_debug,\n', '', content)
content = re.sub(r'            "ollama_model": self\.ollama_model,\n', '', content)

content = content.replace('ollama_temperature', 'ai_temperature')
content = content.replace('ollama_timeout_seconds', 'ai_timeout_seconds')
content = content.replace('ollama_retry_count', 'ai_retry_count')
content = content.replace('ollama_candidate_count', 'ai_candidate_count')
content = content.replace('ollama_max_regeneration_attempts', 'ai_max_regeneration_attempts')
content = content.replace('ollama_quality_retry_enabled', 'ai_quality_retry_enabled')
content = content.replace('ollama_reference_levels_dir', 'reference_levels_dir')
content = content.replace('ollama_context_dir', 'context_dir')
content = content.replace('ollama_max_context_chars', 'ai_max_context_chars')

content = content.replace('"fallback_providers": ["ollama"] if self.fallback_to_ollama and self.ai_provider != "ollama" else [],', '"fallback_providers": [],')

# Remove ollama key validations
content = re.sub(r'    if provider == "ollama".*?\n', '', content)
content = re.sub(r'    has_ollama_key = bool\(config\.ollama_api_key\.strip\(\)\) or \(\n        config\.use_environment_key and environment_api_key_available\("OPENAI_API_KEY"\)\n    \)\n', '', content)
content = re.sub(r'    has_key = has_ollama_key if provider == "ollama" else has_ollama_key\n', '    has_key = has_ollama_key\n', content)
content = re.sub(r'        errors\.append\("Ollama base URL or OLLAMA_HOST is required" if provider == "ollama" else "Ollama base URL or OLLAMA_HOST is required"\)\n', '        errors.append("Ollama base URL or OLLAMA_HOST is required")\n', content)

# Also remove ollama package checks
content = re.sub(r'    if provider == "ollama" and has_key and not config\.enable_local_test_provider and not ollama_package_available\(\):\n        errors\.append\("ollama package is not installed\. Run: pip install ollama"\)\n', '', content)

open('src/gmdgen/gui/app.py', 'w', encoding='utf-8').write(content)
print("done")
