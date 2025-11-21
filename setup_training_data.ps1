# Setup Training Data - Copy from Parent EDON
# Run from edon-cav-engine directory

Write-Host "`n=== Setting Up Training Data ===`n" -ForegroundColor Cyan

# Create external data directory
$externalDir = "data\external\mobiact"
New-Item -ItemType Directory -Force -Path $externalDir | Out-Null

# Copy WESAD CSV (has ACC_x, ACC_y, ACC_z - uppercase)
if (Test-Path "..\data\raw\wesad\wesad_wrist_4hz.csv") {
    Copy-Item "..\data\raw\wesad\wesad_wrist_4hz.csv" "$externalDir\mobiact.csv" -Force
    Write-Host "✓ Copied WESAD CSV to $externalDir\mobiact.csv" -ForegroundColor Green
    Write-Host "  Note: WESAD has ACC_x, ACC_y, ACC_z (uppercase columns)" -ForegroundColor Yellow
    Write-Host "  Parser handles both uppercase and lowercase column names" -ForegroundColor Yellow
} else {
    Write-Host "✗ WESAD CSV not found at ..\data\raw\wesad\wesad_wrist_4hz.csv" -ForegroundColor Red
}

# Also check for the parquet file (100k windows - already processed)
if (Test-Path "..\cav_engine_v3_2_LGBM_2025-11-08\oem_100k_windows.parquet") {
    Write-Host "`n✓ Found processed dataset: oem_100k_windows.parquet" -ForegroundColor Green
    Write-Host "  This can be converted to JSONL if needed" -ForegroundColor Gray
}

Write-Host "`n=== Setup Complete ===`n" -ForegroundColor Cyan

