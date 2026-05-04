Write-Host "Verifying release..."
.\tools\local_ai\verify_final_release.ps1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Verification failed. Stopping package generation."
    exit 1
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$releaseDir = "release_artifacts/gmdgen_ai_editor_$timestamp"
New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null

$zipPath = "$releaseDir/gmdgen_ai_editor_source.zip"

Write-Host "Creating archive $zipPath..."
git archive --format zip --output $zipPath HEAD
Write-Host "Archive created successfully."
