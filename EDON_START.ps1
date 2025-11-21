<#
==========================================
        EDON FULL SYSTEM STARTUP
==========================================
Starts:
  - Virtual environment
  - EDON API (port 8001)
  - Health check loop
  - Humanoid consumer (optional)
  - Home AI consumer (optional)
==========================================
#>

Write-Host "`n=== EDON SYSTEM STARTUP ===" -ForegroundColor Cyan

# -----------------------------
# 1. CONFIGURATION
# -----------------------------
$env:EDON_API_BASE       = "http://127.0.0.1:8001"
$env:EDON_AUTH_ENABLED   = "true"
$env:EDON_API_TOKEN      = "dev-token"
$env:EDON_RELAXED_GUARD  = "1"      # Allow raw windows in dev
$env:EDON_HUMANOID_POLL_SEC = "3.0"
$env:EDON_HUMANOID_DEBUG = "0"      # Set to 1 to see raw JSON

# HOME AI (optional)
# $env:EDON_HOME_ENABLED = "1"
# $env:EDON_HOME_DEVICE_IP = "192.168.1.20"   # Example
# $env:EDON_HOME_API_KEY = "YOUR_KEY"

# -----------------------------
# 2. ACTIVATE VENV
# -----------------------------
Write-Host "[INFO] Activating virtual environment..."
Set-Location $PSScriptRoot
.\.venv\Scripts\Activate.ps1

# -----------------------------
# 3. START EDON API (port 8001)
# -----------------------------
Write-Host "[INFO] Starting EDON API on port 8001..."

Start-Process powershell -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy Bypass",
    "-Command",
    "uvicorn app.main:app --host 0.0.0.0 --port 8001"
) -WindowStyle Minimized

# -----------------------------
# 4. WAIT FOR HEALTH CHECK
# -----------------------------
Write-Host "[INFO] Waiting for EDON API to become healthy..." -ForegroundColor Yellow

$healthy = $false

for ($i = 1; $i -le 20; $i++) {
    try {
        $res = Invoke-RestMethod -Uri "$($env:EDON_API_BASE)/health" -TimeoutSec 2
        if ($res.ok -eq $true) {
            Write-Host "[OK] EDON API is healthy." -ForegroundColor Green
            $healthy = $true
            break
        }
    }
    catch {}
    Start-Sleep -Seconds 1
    Write-Host "[WAIT] Attempt $i ..."
}

if (-not $healthy) {
    Write-Host "[ERROR] EDON API failed to start." -ForegroundColor Red
    exit 1
}

# -----------------------------
# 5. START HUMANOID CONSUMER
# -----------------------------
Write-Host "[INFO] Starting Humanoid Consumer..."
Start-Process powershell -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy Bypass",
    "-File",
    "tools/run_humanoid.ps1"
) -WindowStyle Minimized

# -----------------------------
# 6. START HOME AI (optional)
# -----------------------------
if ($env:EDON_HOME_ENABLED -eq "1") {
    Write-Host "[INFO] Starting Home-AI Consumer..."
    Start-Process powershell -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy Bypass",
        "-File",
        "tools/run_home_ai.ps1"
    ) -WindowStyle Minimized
}

# -----------------------------
# 7. LIVE MONITORING
# -----------------------------
Write-Host "`n=== EDON LIVE STATE MONITOR ===" -ForegroundColor Cyan
Write-Host "Press CTRL+C to exit.`n"

while ($true) {
    try {
        $state = Invoke-RestMethod -Uri "$($env:EDON_API_BASE)/_debug/state" -TimeoutSec 2
        $mode = $state.mode
        if (-not $mode) { $mode = $state.state.state }

        Write-Host "[EDON] mode=$mode cav_smooth=$($state.state.cav_smooth) confidence=$($state.state.confidence)"
    }
    catch {
        Write-Host "[ERR] Unable to fetch state."
    }
    Start-Sleep -Seconds 3
}
