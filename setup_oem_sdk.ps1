<# =====================================================================
 EDON CAV Engine v3.2 - OEM SDK Bootstrap
 Author: Charlie Biggins
 Compatible with older Windows PowerShell
 ====================================================================== #>
[CmdletBinding()]
param(
  [switch]$SkipBackend,
  [switch]$SkipDashboard,
  [switch]$SkipInstall,
  [string]$Python = "",
  [string]$ModelDir = ""
)
$ErrorActionPreference = 'Stop'
Write-Host "Starting EDON OEM SDK setup..."
# Repo root
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
function Resolve-Python {
  param([string]$Preferred)
  if ($Preferred -and (Test-Path $Preferred)) { return $Preferred }
  # Try Windows py launcher first (most reliable)
  $pyPath = ""
  try { $pyPath = (& py -3 -c "import sys; print(sys.executable)" 2>$null) } catch {}
  if ($pyPath) {
    $pyPath = $pyPath.Trim()
    if (Test-Path $pyPath) { return $pyPath }
  }
  # Fallback to python on PATH
  $gc = $null
  try { $gc = Get-Command python -ErrorAction SilentlyContinue } catch {}
  if ($null -ne $gc -and $gc.Source) { return $gc.Source }
  throw "Python not found. Pass -Python C:\Path\to\python.exe"
}
$PythonExe = Resolve-Python -Preferred $Python
Write-Host "Using Python: $PythonExe"
# Create venv if missing
$Activate = ".\.venv\Scripts\Activate.ps1"
if (-not (Test-Path $Activate)) {
  Write-Host "Creating virtual environment (.venv)"
  & $PythonExe -m venv .venv
}
# Activate venv
. $Activate
Write-Host "Virtual environment activated"
# Install deps (optional)
if (-not $SkipInstall) {
  Write-Host "Upgrading pip and installing requirements..."
  python -m pip install --upgrade pip
  if (-not (Test-Path ".\requirements.txt")) { throw "requirements.txt not found in $Root" }
  pip install -r .\requirements.txt
} else {
  Write-Host "Skipping dependency install (-SkipInstall)"
}
# Env vars
$env:PYTHONPATH = $Root
if ($ModelDir) {
  $env:EDON_MODEL_DIR = $ModelDir
} elseif (-not $env:EDON_MODEL_DIR) {
  $env:EDON_MODEL_DIR = "cav_engine_v3_2_LGBM_2025-11-08"
}
Write-Host "PYTHONPATH=$env:PYTHONPATH"
Write-Host "EDON_MODEL_DIR=$env:EDON_MODEL_DIR"
# Launch services
if (-not $SkipBackend) {
  Start-Job -Name edon_api -ScriptBlock { uvicorn app.main:app --reload --port 8000 } | Out-Null
  Write-Host "FastAPI backend → http://localhost:8000"
} else {
  Write-Host "Skipping backend (-SkipBackend)"
}
if (-not $SkipDashboard) {
  Start-Job -Name edon_ui -ScriptBlock { streamlit run tools\cav_dashboard.py } | Out-Null
  Write-Host "Dashboard → http://localhost:8501"
} else {
  Write-Host "Skipping dashboard (-SkipDashboard)"
}
Write-Host ""
Write-Host "EDON OEM SDK setup complete."
Write-Host "Stop jobs with: Get-Job | Stop-Job; Get-Job | Remove-Job"
