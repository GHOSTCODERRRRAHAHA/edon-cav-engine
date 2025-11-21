# Copy All Required Datasets from Parent EDON Folder
# Run from edon-cav-engine directory

Write-Host "`n=== Copying All Required Datasets ===`n" -ForegroundColor Cyan

$parent = ".."
$ErrorActionPreference = "Continue"

# 1. Copy WESAD data (needed for evaluation)
Write-Host "1. Copying WESAD data..." -ForegroundColor Yellow
$wesadTargetDir = "data\raw\wesad"
New-Item -ItemType Directory -Force -Path $wesadTargetDir | Out-Null

$wesadFiles = @(
    "wesad_wrist_4hz.csv",
    "wesad_wrist_4hz.parquet",
    "wesad_wrist_4hz_clean.csv"
)

foreach ($file in $wesadFiles) {
    if (Test-Path "$parent\data\raw\wesad\$file") {
        Copy-Item "$parent\data\raw\wesad\$file" "$wesadTargetDir\" -Force
        Write-Host "   ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "   - $file not found (optional)" -ForegroundColor Gray
    }
}

# 2. Verify unified datasets exist
Write-Host "`n2. Checking unified datasets..." -ForegroundColor Yellow
$unifiedDir = "data\unified"
if (Test-Path "$unifiedDir\all_v10.jsonl") {
    Write-Host "   ✓ all_v10.jsonl exists" -ForegroundColor Green
} else {
    Write-Host "   ✗ all_v10.jsonl missing" -ForegroundColor Red
}

if (Test-Path "$unifiedDir\wisdm.jsonl") {
    Write-Host "   ✓ wisdm.jsonl exists" -ForegroundColor Green
} else {
    Write-Host "   ✗ wisdm.jsonl missing" -ForegroundColor Red
}

if (Test-Path "$unifiedDir\mobiact.jsonl") {
    Write-Host "   ✓ mobiact.jsonl exists" -ForegroundColor Green
} else {
    Write-Host "   ✗ mobiact.jsonl missing" -ForegroundColor Red
}

# 3. Check external datasets
Write-Host "`n3. Checking external datasets..." -ForegroundColor Yellow
if (Test-Path "data\external\mobiact\mobiact.csv") {
    Write-Host "   ✓ mobiact.csv exists" -ForegroundColor Green
} else {
    Write-Host "   - mobiact.csv not found (optional)" -ForegroundColor Gray
}

Write-Host "`n=== Summary ===" -ForegroundColor Cyan
Write-Host "WESAD data location: $wesadTargetDir\wesad_wrist_4hz.csv" -ForegroundColor Green
Write-Host "Unified datasets: $unifiedDir\" -ForegroundColor Green
Write-Host "`n=== Copy Complete ===`n" -ForegroundColor Cyan

