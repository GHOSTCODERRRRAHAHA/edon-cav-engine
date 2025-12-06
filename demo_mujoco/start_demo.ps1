# PowerShell script to start the EDON MuJoCo demo

Write-Host "EDON MuJoCo Stability Demo" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan
Write-Host ""

# Disable adaptive memory for consistent demo performance
# Adaptive memory can cause variability in zero-shot mode
# For demos, we want consistent base v8 policy performance
$env:EDON_DISABLE_ADAPTIVE_MEMORY = "1"
Write-Host "Adaptive memory disabled for consistent demo performance" -ForegroundColor Yellow
Write-Host ""

# Check if EDON server is running
Write-Host "Checking EDON server..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 2 -ErrorAction Stop
    Write-Host "✓ EDON server is running" -ForegroundColor Green
} catch {
    Write-Host "✗ EDON server not found at http://localhost:8000" -ForegroundColor Red
    Write-Host "  Please start the EDON server first:" -ForegroundColor Yellow
    Write-Host "  python -m app.main" -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") {
        exit 1
    }
}

Write-Host ""
Write-Host "Starting demo..." -ForegroundColor Yellow
Write-Host "UI will be available at: http://localhost:8080" -ForegroundColor Cyan
Write-Host ""
Write-Host "Note: Adaptive memory is disabled for consistent performance" -ForegroundColor Gray
Write-Host ""

# Run the demo
python run_demo.py
