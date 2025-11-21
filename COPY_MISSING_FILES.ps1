# Copy Missing Files from Parent EDON Folder
# Run this from edon-cav-engine directory

$parent = ".."
$ErrorActionPreference = "Continue"

Write-Host "`n=== Copying Missing Files from Parent EDON ===`n" -ForegroundColor Cyan

# 1. Copy model files (CRITICAL)
Write-Host "1. Copying model files..." -ForegroundColor Yellow
$modelDir = "cav_engine_v3_2_LGBM_2025-11-08"
if (Test-Path "$parent\$modelDir\cav_state_v3_2.joblib") {
    Copy-Item "$parent\$modelDir\cav_state_v3_2.joblib" "$modelDir\" -Force
    Write-Host "   ✓ cav_state_v3_2.joblib" -ForegroundColor Green
} else {
    Write-Host "   ✗ cav_state_v3_2.joblib not found in parent" -ForegroundColor Red
}

if (Test-Path "$parent\$modelDir\cav_state_scaler_v3_2.joblib") {
    Copy-Item "$parent\$modelDir\cav_state_scaler_v3_2.joblib" "$modelDir\" -Force
    Write-Host "   ✓ cav_state_scaler_v3_2.joblib" -ForegroundColor Green
} else {
    Write-Host "   ✗ cav_state_scaler_v3_2.joblib not found in parent" -ForegroundColor Red
}

# 2. Copy tools
Write-Host "`n2. Copying tools..." -ForegroundColor Yellow
if (Test-Path "$parent\tools\oem_dashboard.py") {
    Copy-Item "$parent\tools\oem_dashboard.py" "tools\" -Force
    Write-Host "   ✓ oem_dashboard.py" -ForegroundColor Green
} else {
    Write-Host "   ✗ oem_dashboard.py not found" -ForegroundColor Yellow
}

if (Test-Path "$parent\tools\manifest_utils.py") {
    Copy-Item "$parent\tools\manifest_utils.py" "tools\" -Force
    Write-Host "   ✓ manifest_utils.py" -ForegroundColor Green
} else {
    Write-Host "   ✗ manifest_utils.py not found" -ForegroundColor Yellow
}

# 3. Copy app files that are imported
Write-Host "`n3. Copying app files..." -ForegroundColor Yellow
if (Test-Path "$parent\app\edge_bridge.py") {
    Copy-Item "$parent\app\edge_bridge.py" "app\" -Force
    Write-Host "   ✓ edge_bridge.py" -ForegroundColor Green
} else {
    Write-Host "   ✗ edge_bridge.py not found" -ForegroundColor Yellow
}

# 4. Copy config
Write-Host "`n4. Copying config..." -ForegroundColor Yellow
if (Test-Path "$parent\config\config.yaml") {
    New-Item -ItemType Directory -Force -Path "config" | Out-Null
    Copy-Item "$parent\config\config.yaml" "config\" -Force
    Write-Host "   ✓ config.yaml" -ForegroundColor Green
} else {
    Write-Host "   ✗ config.yaml not found" -ForegroundColor Yellow
}

# 5. Copy WESAD data (needed for evaluation)
Write-Host "`n5. Copying WESAD data..." -ForegroundColor Yellow
$wesadTargetDir = "data\raw\wesad"
New-Item -ItemType Directory -Force -Path $wesadTargetDir | Out-Null

if (Test-Path "$parent\data\raw\wesad\wesad_wrist_4hz.csv") {
    Copy-Item "$parent\data\raw\wesad\wesad_wrist_4hz.csv" "$wesadTargetDir\" -Force
    Write-Host "   ✓ wesad_wrist_4hz.csv" -ForegroundColor Green
} else {
    Write-Host "   ✗ wesad_wrist_4hz.csv not found" -ForegroundColor Yellow
}

if (Test-Path "$parent\data\raw\wesad\wesad_wrist_4hz.parquet") {
    Copy-Item "$parent\data\raw\wesad\wesad_wrist_4hz.parquet" "$wesadTargetDir\" -Force
    Write-Host "   ✓ wesad_wrist_4hz.parquet" -ForegroundColor Green
}

Write-Host "`n=== Copy Complete ===`n" -ForegroundColor Cyan
Write-Host "All necessary files copied to edon-cav-engine folder" -ForegroundColor Green

