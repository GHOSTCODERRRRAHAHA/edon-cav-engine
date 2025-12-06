# Start EDON Core Server for Testing
# This starts the REST API server directly (no Docker needed)

Write-Host "Starting EDON Core Server..." -ForegroundColor Green

# Activate virtual environment
$venvPath = Join-Path $PSScriptRoot ".venv"
if (Test-Path $venvPath) {
    Write-Host "[INFO] Activating virtual environment..." -ForegroundColor Cyan
    & "$venvPath\Scripts\Activate.ps1"
} else {
    Write-Host "[WARN] Virtual environment not found at $venvPath" -ForegroundColor Yellow
    Write-Host "[INFO] Continuing with system Python..." -ForegroundColor Yellow
}

# Check if server is already running
try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:8002/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        Write-Host "[OK] EDON Core server is already running on port 8002" -ForegroundColor Yellow
        Write-Host "You can test it with: curl http://127.0.0.1:8002/health" -ForegroundColor Cyan
        exit 0
    }
} catch {
    # Server not running, continue
}

# Start server
Write-Host "[INFO] Starting REST API server on port 8002..." -ForegroundColor Cyan
Write-Host "[INFO] Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

python -m uvicorn app.main:app --host 127.0.0.1 --port 8002

