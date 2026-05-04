param (
    [string]$BaseModel = "qwen2.5-coder:7b",
    [string]$Alias = "gmdgen-coder-final"
)

Write-Host "Creating alias '$Alias' for model '$BaseModel'..."

# if 7B is too heavy, recommend fallback
Write-Host "(If 7B is too heavy for your system, you can use qwen2.5-coder:3b instead)"

$modelfileContent = "FROM $BaseModel"
$modelfilePath = [System.IO.Path]::GetTempFileName()
Set-Content -Path $modelfilePath -Value $modelfileContent

try {
    ollama create $Alias -f $modelfilePath
    Write-Host "Alias created successfully!"
} finally {
    Remove-Item -Path $modelfilePath -ErrorAction SilentlyContinue
}
