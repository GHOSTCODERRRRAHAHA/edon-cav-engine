#!/usr/bin/env pwsh
"""
EDON CAV API Runner - PowerShell Script

Activates virtual environment and starts FastAPI server.
"""

$ErrorActionPreference = "Stop"

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Check if venv exists
$VenvPath = Join-Path $ScriptDir "venv"
if (-not (Test-Path $VenvPath)) {
    Write-Host "[ERROR] Virtual environment not found at: $VenvPath" -ForegroundColor Red
    Write-Host "Please create it first: python -m venv venv" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
if (Test-Path $ActivateScript) {
    Write-Host "[INFO] Activating virtual environment..." -ForegroundColor Green
    & $ActivateScript
} else {
    Write-Host "[ERROR] Activation script not found: $ActivateScript" -ForegroundColor Red
    exit 1
}

# Check if Python is available
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Host "[ERROR] Python executable not found: $PythonExe" -ForegroundColor Red
    exit 1
}

# Check if uvicorn is installed
Write-Host "[INFO] Checking dependencies..." -ForegroundColor Green
$UvicornCheck = & $PythonExe -c "import uvicorn; print('OK')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] uvicorn not installed. Installing requirements..." -ForegroundColor Yellow
    & $PythonExe -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install requirements" -ForegroundColor Red
        exit 1
    }
}

# Start FastAPI server
Write-Host "[INFO] Starting FastAPI server..." -ForegroundColor Green
Write-Host "[INFO] Server will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "[INFO] API docs available at: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "[INFO] Press CTRL+C to stop the server" -ForegroundColor Yellow
Write-Host ""

& $PythonExe -m uvicorn app.main:app --host 0.0.0.0 --port 8000

