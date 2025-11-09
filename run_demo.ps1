# ============================================================
# EDON CAV Engine v3.2 - Quick Demo Runner
# ============================================================
# Runs demo_infer.py with proper Python environment setup
#

[CmdletBinding()]
param(
    [switch]$Dashboard,    # Run dashboard instead of demo
    [string]$Python = "python"  # Python executable path
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "EDON CAV Engine v3.2 — Demo Runner" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = & $Python --version 2>&1
    Write-Host "Using: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.8+ or set -Python parameter." -ForegroundColor Red
    exit 1
}

# Check if demo_infer.py exists
if (-not (Test-Path "demo_infer.py")) {
    Write-Host "❌ demo_infer.py not found in current directory." -ForegroundColor Red
    Write-Host "   Make sure you're in the SDK root directory." -ForegroundColor Yellow
    exit 1
}

# Check if models exist
$modelsDir = "models"
if (-not (Test-Path $modelsDir)) {
    Write-Host "⚠️  Warning: models/ directory not found." -ForegroundColor Yellow
    Write-Host "   The script will try to find models using robust discovery." -ForegroundColor Yellow
}

# Run demo or dashboard
if ($Dashboard) {
    Write-Host "Starting Streamlit dashboard..." -ForegroundColor Cyan
    Write-Host ""
    
    # Check if streamlit is installed
    try {
        & $Python -m streamlit --version 2>&1 | Out-Null
    } catch {
        Write-Host "⚠️  Streamlit not found. Installing..." -ForegroundColor Yellow
        & $Python -m pip install streamlit --quiet
    }
    
    # Check if cav_dashboard.py exists
    $dashboardPath = "cav_dashboard.py"
    if (-not (Test-Path $dashboardPath)) {
        Write-Host "❌ cav_dashboard.py not found." -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Opening dashboard at http://localhost:8501" -ForegroundColor Green
    Write-Host ""
    & $Python -m streamlit run $dashboardPath
} else {
    Write-Host "Running demo inference..." -ForegroundColor Cyan
    Write-Host ""
    & $Python demo_infer.py
}

Write-Host ""
Write-Host "✅ Done" -ForegroundColor Green

