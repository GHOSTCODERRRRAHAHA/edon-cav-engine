# Copy WESAD Data from Parent EDON Folder
# Run from edon-cav-engine directory

Write-Host "`n=== Copying WESAD Data from Parent EDON ===`n" -ForegroundColor Cyan

$parent = ".."
$ErrorActionPreference = "Continue"

# Create data/raw/wesad directory structure
$targetDir = "data\raw\wesad"
New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
Write-Host "Created directory: $targetDir" -ForegroundColor Green

# Copy WESAD CSV file (needed for evaluation)
if (Test-Path "$parent\data\raw\wesad\wesad_wrist_4hz.csv") {
    Copy-Item "$parent\data\raw\wesad\wesad_wrist_4hz.csv" "$targetDir\" -Force
    Write-Host "✓ Copied wesad_wrist_4hz.csv" -ForegroundColor Green
} else {
    Write-Host "✗ wesad_wrist_4hz.csv not found in parent" -ForegroundColor Red
}

# Also copy parquet version if it exists
if (Test-Path "$parent\data\raw\wesad\wesad_wrist_4hz.parquet") {
    Copy-Item "$parent\data\raw\wesad\wesad_wrist_4hz.parquet" "$targetDir\" -Force
    Write-Host "✓ Copied wesad_wrist_4hz.parquet" -ForegroundColor Green
}

# Copy cleaned version if it exists
if (Test-Path "$parent\data\raw\wesad\wesad_wrist_4hz_clean.csv") {
    Copy-Item "$parent\data\raw\wesad\wesad_wrist_4hz_clean.csv" "$targetDir\" -Force
    Write-Host "✓ Copied wesad_wrist_4hz_clean.csv" -ForegroundColor Green
}

Write-Host "`n=== Copy Complete ===`n" -ForegroundColor Cyan
Write-Host "WESAD data is now available at: $targetDir\wesad_wrist_4hz.csv" -ForegroundColor Green

