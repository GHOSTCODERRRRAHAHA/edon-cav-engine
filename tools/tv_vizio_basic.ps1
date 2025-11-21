# tools\tv_vizio_basic.ps1
# Minimal Vizio SmartCast control (based on Vizio_SmartCast_API style)
# Uses ONLY key commands: volume up/down, no broken /state/volume/master.

# -----------------------------
# Trust all certs (if not set)
# -----------------------------
if (-not ('TrustAllCertsPolicy' -as [type])) {
    Add-Type @"
using System.Net;
using System.Security.Cryptography.X509Certificates;
public class TrustAllCertsPolicy : ICertificatePolicy {
    public bool CheckValidationResult(
        ServicePoint srvPoint,
        X509Certificate certificate,
        WebRequest request,
        int certificateProblem) {
        return true;
    }
}
"@
    [System.Net.ServicePointManager]::CertificatePolicy = New-Object TrustAllCertsPolicy
}

# -----------------------------
# Helpers
# -----------------------------

function Invoke-VizioKey {
    param(
        [Parameter(Mandatory)] [int]$Codeset,
        [Parameter(Mandatory)] [int]$Code,
        [int]$Times   = 1,
        [int]$DelayMs = 100
    )

    if (-not $env:VIZIO_BASE -or -not $env:VIZIO_AUTH) {
        throw "VIZIO_BASE or VIZIO_AUTH is not set. Set them before using Invoke-VizioKey."
    }

    $uri  = "$($env:VIZIO_BASE.TrimEnd('/'))/key_command/"
    $body = @{ KEYLIST = @(@{ CODESET = $Codeset; CODE = $Code; ACTION = 'KEYPRESS' }) } |
        ConvertTo-Json -Compress

    for ($i = 0; $i -lt $Times; $i++) {
        Invoke-RestMethod -Uri $uri `
                          -Headers @{ AUTH = $env:VIZIO_AUTH } `
                          -Method PUT `
                          -Body $body `
                          -ContentType 'application/json' | Out-Null
        Start-Sleep -Milliseconds $DelayMs
    }
}

# Track our best guess of current volume (0–100)
if (-not (Get-Variable -Name CurrentVolume -Scope Script -ErrorAction SilentlyContinue)) {
    $script:CurrentVolume = $null
}

# 5 = remote control codeset for volume/channel/etc on most SmartCast TVs
# CODE 1 = VOL UP, CODE 0 = VOL DOWN

function TV-VolUp {
    param([int]$Steps = 1)
    Invoke-VizioKey -Codeset 5 -Code 1 -Times $Steps -DelayMs 80
}

function TV-VolDown {
    param([int]$Steps = 1)
    Invoke-VizioKey -Codeset 5 -Code 0 -Times $Steps -DelayMs 80
}

# -----------------------------
# Exact volume setter (no auto-zero)
# -----------------------------
function Set-TVVolumeExact {
    param(
        [Parameter(Mandatory)][int]$Target
    )

    # Clamp to a sane human range (0–100); adjust if your TV max is lower.
    $Target = [Math]::Max(0, [Math]::Min(100, $Target))

    # First-time use: DO NOT slam to 0.
    # Just assume current TV volume is already at Target and sync baseline.
    if ($script:CurrentVolume -eq $null) {
        Write-Host "[TV] First-time set: assuming TV is already at $Target; syncing baseline only."
        $script:CurrentVolume = $Target
        return
    }

    # Normal path: move relative from our last known volume
    $delta = $Target - $script:CurrentVolume

    if ($delta -eq 0) {
        Write-Host "[TV] Volume already at $Target (shadow); no change needed."
        return
    }

    if ($delta -gt 0) {
        Write-Host "[TV] Raising volume from $($script:CurrentVolume) → $Target (+$delta)."
        TV-VolUp -Steps $delta
    } else {
        $stepsDown = -1 * $delta
        Write-Host "[TV] Lowering volume from $($script:CurrentVolume) → $Target (-$stepsDown)."
        TV-VolDown -Steps $stepsDown
    }

    $script:CurrentVolume = $Target
}

# -----------------------------
# Manual sync helper
# -----------------------------
function Sync-TVVolumeManual {
    param(
        [Parameter(Mandatory)][int]$Current
    )

    $script:CurrentVolume = [Math]::Max(0, [Math]::Min(100, $Current))
    Write-Host "[TV] Synced internal volume to $($script:CurrentVolume)."
}
