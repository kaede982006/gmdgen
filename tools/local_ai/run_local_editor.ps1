try {
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -Method Get -ErrorAction Stop
} catch {
    Write-Host "Ollama server is not running or available."
    Write-Host "Please run the following commands:"
    Write-Host "ollama serve"
    Write-Host "ollama pull qwen2.5-coder:7b"
    Write-Host ".\tools\local_ai\create_ollama_model_alias.ps1"
    exit 1
}

python -m gmdgen
