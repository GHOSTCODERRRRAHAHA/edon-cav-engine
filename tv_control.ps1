# tv_control.ps1
# TV volume control helpers for EDON → TV loop

# In-memory "calibration" (starting point)
if (-not $global:TVCal) {
    $global:TVCal = @{
        Step    = 1      # volume change per press (tweak if needed)
        Current = 15     # set this to your TV's actual current volume
    }
}

function Set-TVVolume {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Target
    )

    # Clamp volume to a sane range
    if ($Target -lt 0) { $Target = 0 }
    if ($Target -gt 100) { $Target = 100 }

    $step    = [int]$global:TVCal.Step
    $current = [int]$global:TVCal.Current
    $delta   = $Target - $current

    if ($delta -eq 0) {
        Write-Host "[TV] Volume already at $current; no action needed." -ForegroundColor DarkGray
        return
    }

    $presses = [math]::Abs([int]([double]$delta / [double]$step))
    if ($presses -eq 0) { $presses = 1 }

    $direction = if ($delta -gt 0) { "UP" } else { "DOWN" }

    Write-Host "[TV] Changing volume from $current to $Target (~$presses $direction presses)" -ForegroundColor Green

    # ============================================================
    # REAL TV CONTROL HOOK
    # ============================================================
    # If you already have a function like Send-TVKey, Send-CECCommand,
    # or an IR blaster cmdlet, plug it in here.
    #
    # Example if you had:
    #   function Send-TVKey { param([string]$Key) ... }
    # You’d do:
    #
    # if (Get-Command Send-TVKey -ErrorAction SilentlyContinue) {
    #     for ($i = 0; $i -lt $presses; $i++) {
    #         Send-TVKey -Key $direction   # or 'VOLUP'/'VOLDOWN'
    #         Start-Sleep -Milliseconds 150
    #     }
    # } else {
    #     # fallback to SIM
    # }
    #
    # For now we keep the SIM as a fallback.

    if (Get-Command Send-TVKey -ErrorAction SilentlyContinue) {
        # 🔥 Replace 'VOLUP'/'VOLDOWN' if your function expects something else
        $key = if ($direction -eq "UP") { "VOLUP" } else { "VOLDOWN" }

        for ($i = 0; $i -lt $presses; $i++) {
            Write-Host "  [REAL] Sending $key" -ForegroundColor Yellow
            Send-TVKey -Key $key
            Start-Sleep -Milliseconds 150
        }
    }
    else {
        # Fallback SIMULATION if no real command exists yet
        for ($i = 0; $i -lt $presses; $i++) {
            Write-Host "  [SIM] Press $direction" -ForegroundColor Magenta
            Start-Sleep -Milliseconds 100
        }
    }

    # Update our remembered current volume
    $global:TVCal.Current = $Target
}

function Get-TVVolume {
    return [int]$global:TVCal.Current
}
