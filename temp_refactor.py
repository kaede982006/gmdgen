import os
import re

patterns = [
    (r'ollama_api_key=', 'ollama_base_url='),
    (r'"ollama_api_key"', '"ollama_base_url"'),
    (r"'ollama_api_key'", "'ollama_base_url'"),
    (r'Ollama API key is required', 'Ollama base URL or OLLAMA_HOST is required'),
    (r'ollama_api_key_missing', 'ollama_base_url_missing'),
    (r'\[REDACTED_OLLAMA_API_KEY\]', '[REDACTED_OLLAMA_HOST]'),
    (r'OLLAMA_API_KEY', 'OLLAMA_HOST'),
]

all_files = [
    'src/gmdgen/ai/fine_tune_export.py',
    'src/gmdgen/errors.py',
    'src/gmdgen/eval/live_ollama_eval.py',
    'src/gmdgen/gui/app.py',
    'tests/strip_app.py',
    'tests/strip_openai_app.py',
    'tests/strip_openai_app2.py',
    'tests/test_dataset_index.py',
    'tests/test_error_handling.py',
    'tests/test_feedback.py',
    'tests/test_fine_tune_export.py',
    'tests/test_gui_config_contract.py',
    'tests/test_gui_worker.py',
    'tests/test_learning_feature_extractor.py',
    'tests/test_learning_store.py',
    'tests/test_preference_export.py',
]

changed_files = []

for file_path in all_files:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content
    for pattern, replacement in patterns:
        new_content = re.sub(pattern, replacement, new_content)
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        changed_files.append(file_path)

print('Changed files:')
for f in changed_files:
    print(f)
