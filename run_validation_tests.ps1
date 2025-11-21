# Run All 3 Validation Tests
$ErrorActionPreference = "Continue"

Write-Host "`n=== EDON Pilot Validation Tests ===`n" -ForegroundColor Cyan

$edonDir = "C:\Users\cjbig\Desktop\EDON\edon-cav-engine"
Set-Location $edonDir

# Check if server is running
Write-Host "Checking if server is running..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8001/health" -Method Get -TimeoutSec 2 -ErrorAction Stop
    Write-Host "[OK] Server is running" -ForegroundColor Green
} catch {
    Write-Host "[FAIL] Server is not running. Starting server..." -ForegroundColor Red
    Start-Process -FilePath ".\venv\Scripts\python.exe" -ArgumentList "-m","uvicorn","app.main:app","--host","127.0.0.1","--port","8001" -WindowStyle Hidden
    Write-Host "Waiting 8 seconds for server to start..." -ForegroundColor Yellow
    Start-Sleep -Seconds 8
    
    # Verify server started
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:8001/health" -Method Get -TimeoutSec 2 -ErrorAction Stop
        Write-Host "[OK] Server started successfully" -ForegroundColor Green
    } catch {
        Write-Host "[FAIL] Server failed to start. Please start manually." -ForegroundColor Red
        exit 1
    }
}

# Test 1: Model Info
Write-Host "`n=== TEST 1: Model Info ===" -ForegroundColor Cyan
try {
    $modelInfo = Invoke-RestMethod -Uri "http://127.0.0.1:8001/models/info" -Method Get
    Write-Host "Model Name: $($modelInfo.name)" -ForegroundColor Green
    Write-Host "Model Hash: $($modelInfo.sha256)" -ForegroundColor Green
    Write-Host "Features: $($modelInfo.features)" -ForegroundColor Green
    Write-Host "Window: $($modelInfo.window)" -ForegroundColor Green
    Write-Host "PCA Dims: $($modelInfo.pca_dim)" -ForegroundColor Green
    
    # Create reports directory if it doesn't exist
    New-Item -ItemType Directory -Force -Path "reports" | Out-Null
    $modelInfo | ConvertTo-Json -Depth 10 | Out-File -FilePath "reports\model_info.json" -Encoding UTF8
    Write-Host "[OK] Model info saved to reports\model_info.json" -ForegroundColor Green
} catch {
    Write-Host "[FAIL] Error getting model info: $_" -ForegroundColor Red
}

# Test 2: Evaluation
Write-Host "`n=== TEST 2: Evaluation (WESAD Ground Truth) ===" -ForegroundColor Cyan
# Try local path first, then parent folder
$wesadPath = "data\raw\wesad\wesad_wrist_4hz.csv"
if (-not (Test-Path $wesadPath)) {
    $wesadPath = "..\data\raw\wesad\wesad_wrist_4hz.csv"
}
if (Test-Path $wesadPath) {
    Write-Host "Running evaluation on 50 windows (limited for speed)..." -ForegroundColor Yellow
    & .\venv\Scripts\python.exe tools\eval_wesad.py --data $wesadPath --api http://127.0.0.1:8001 --output reports\last_eval.json --limit 50
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Evaluation complete" -ForegroundColor Green
        if (Test-Path "reports\last_eval.json") {
            $eval = Get-Content "reports\last_eval.json" | ConvertFrom-Json
            Write-Host "  Accuracy: $($eval.accuracy)" -ForegroundColor Cyan
            Write-Host "  AUROC: $($eval.auroc)" -ForegroundColor Cyan
        }
    } else {
        Write-Host "[FAIL] Evaluation failed" -ForegroundColor Red
    }
} else {
    Write-Host "[FAIL] WESAD data not found at $wesadPath" -ForegroundColor Red
}

# Test 3: Load Test
Write-Host "`n=== TEST 3: Load Test (50 requests, 3 windows each) ===" -ForegroundColor Cyan
& .\venv\Scripts\python.exe tools\load_test.py --url http://127.0.0.1:8001/oem/cav/batch --requests 50 --windows 3 --concurrent 5
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Load test passed (p95 less than 120ms)" -ForegroundColor Green
} else {
    Write-Host "[FAIL] Load test failed or p95 greater than 120ms" -ForegroundColor Red
}

Write-Host "`n=== Validation Tests Complete ===`n" -ForegroundColor Cyan
Write-Host "Results saved to:" -ForegroundColor Yellow
Write-Host "  - reports\model_info.json" -ForegroundColor Gray
Write-Host "  - reports\last_eval.json" -ForegroundColor Gray

