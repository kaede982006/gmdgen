try {
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -Method Get -ErrorAction Stop
} catch {
    Write-Host "Ollama server is not running or available. Skipping smoke test."
    exit 0
}

Write-Host "Running minimal provider smoke test..."
python -c "
from gmdgen.ai.factory import create_ai_provider_from_config
provider = create_ai_provider_from_config({'ai_provider': 'ollama', 'ollama_model': 'qwen2.5-coder:7b'})
print('Provider created successfully.')
"
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "Smoke test passed."
