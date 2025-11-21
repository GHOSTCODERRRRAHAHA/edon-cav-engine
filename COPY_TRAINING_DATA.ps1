# Copy Training Data from Parent EDON Folder
# Run this from edon-cav-engine directory

$parent = ".."
$ErrorActionPreference = "Continue"

Write-Host "`n=== Copying Training Data from Parent EDON ===`n" -ForegroundColor Cyan

# Option 1: Copy WESAD CSV for MobiAct parser
Write-Host "1. Setting up WESAD data for MobiAct parser..." -ForegroundColor Yellow
$externalDir = "data\external\mobiact"
New-Item -ItemType Directory -Force -Path $externalDir | Out-Null

if (Test-Path "$parent\data\raw\wesad\wesad_wrist_4hz.csv") {
    Copy-Item "$parent\data\raw\wesad\wesad_wrist_4hz.csv" "$externalDir\mobiact.csv" -Force
    Write-Host "   ✓ Copied wesad_wrist_4hz.csv to $externalDir\mobiact.csv" -ForegroundColor Green
} else {
    Write-Host "   ✗ wesad_wrist_4hz.csv not found in parent" -ForegroundColor Red
}

# Option 2: Copy the processed parquet file (100k windows)
Write-Host "`n2. Checking for processed dataset..." -ForegroundColor Yellow
if (Test-Path "$parent\cav_engine_v3_2_LGBM_2025-11-08\oem_100k_windows.parquet") {
    Write-Host "   ✓ Found oem_100k_windows.parquet in parent" -ForegroundColor Green
    Write-Host "   (Can be converted to JSONL if needed)" -ForegroundColor Gray
} else {
    Write-Host "   ✗ oem_100k_windows.parquet not found" -ForegroundColor Yellow
}

# Option 3: Check if we already have real_wesad.csv
Write-Host "`n3. Checking local sensor data..." -ForegroundColor Yellow
if (Test-Path "sensors\real_wesad.csv") {
    Write-Host "   ✓ Found sensors\real_wesad.csv (already here)" -ForegroundColor Green
} else {
    Write-Host "   ✗ sensors\real_wesad.csv not found" -ForegroundColor Yellow
}

Write-Host "`n=== Copy Complete ===`n" -ForegroundColor Cyan
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Run .\build_v1.ps1 to parse and train" -ForegroundColor White
Write-Host "  2. Or create synthetic data in data\unified\all_v10.jsonl" -ForegroundColor White

