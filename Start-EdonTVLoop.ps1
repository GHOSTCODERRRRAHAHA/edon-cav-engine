param(
    [string]$BaseUrl = "http://127.0.0.1:8001",
    [int]$PollSeconds = 10
)

Write-Host ""
Write-Host "▶ Starting EDON → TV control loop..." -ForegroundColor Cyan
Write-Host "   Base URL: $BaseUrl" -ForegroundColor DarkGray
Write-Host "   Poll every $PollSeconds seconds" -ForegroundColor DarkGray
Write-Host ""

# -------------------------------
# Dedicated HttpClient for EDON
# -------------------------------
Add-Type -AssemblyName System.Net.Http
if (-not $script:EdonHttpClient) {
    $script:EdonHttpClient = [System.Net.Http.HttpClient]::new()
}

# Auth header (optional)
$authHeader = $null
if ($env:EDON_AUTH_ENABLED -eq "true" -and $env:EDON_API_TOKEN) {
    $authHeader = $env:EDON_API_TOKEN
    Write-Host "[AUTH] Using bearer token for EDON" -ForegroundColor DarkGray
}

# -------------------------------
# State → target volume mapping
# -------------------------------
function Get-TargetVolumeForState {
    param(
        [string]$state,
        [double]$z
    )

    switch ($state) {
        "overload"    { return 4 }   # very low / almost mute
        "focus"       { return 8 }   # low background
        "balanced"    { return 14 }  # normal
        "restorative" { return 10 }  # soft ambient
        default       { return 12 }  # fallback
    }
}

# Remember last state + volume so we don't spam the TV
$lastState  = $null
$lastVolume = $null

while ($true) {
    try {
        $url = "$BaseUrl/_debug/state"

        $req = [System.Net.Http.HttpRequestMessage]::new(
            [System.Net.Http.HttpMethod]::Get,
            $url
        )

        if ($authHeader) {
            # Set Authorization: Bearer <token>
            $req.Headers.Authorization =
                [System.Net.Http.Headers.AuthenticationHeaderValue]::new("Bearer", $authHeader)
        }

        $resp = $script:EdonHttpClient.SendAsync($req).Result
        if (-not $resp.IsSuccessStatusCode) {
            Write-Host ("[EDON] HTTP {0} from /_debug/state" -f [int]$resp.StatusCode) -ForegroundColor Yellow
            Start-Sleep -Seconds $PollSeconds
            continue
        }

        $json = $resp.Content.ReadAsStringAsync().Result
        $res  = $json | ConvertFrom-Json

        if (-not $res.ok) {
            Write-Host "[EDON] /_debug/state returned ok=false; skipping this cycle" -ForegroundColor Yellow
        } else {
            $stateObj = $res.state
            if (-not $stateObj) {
                Write-Host "[EDON] No state object yet (engine warming up?)" -ForegroundColor Yellow
            } else {
                $stateName = $stateObj.state
                $cavRaw    = $stateObj.cav_raw
                $cavSmooth = $stateObj.cav_smooth
                $conf      = $stateObj.confidence
                $z         = $stateObj.z_cav

                if (-not $stateName) {
                    Write-Host "[EDON] State missing in payload; raw: $($stateObj | ConvertTo-Json -Depth 5)" -ForegroundColor Yellow
                } else {
                    $targetVol = Get-TargetVolumeForState -state $stateName -z $z

                    $changedState = ($stateName -ne $lastState)
                    $changedVol   = ($null -eq $lastVolume -or [int]$targetVol -ne [int]$lastVolume)

                    Write-Host ("[EDON] state={0} z={1} cav={2} conf={3} → targetVol={4}" -f `
                        $stateName, $z, $cavSmooth, $conf, $targetVol) -ForegroundColor Cyan

                    if ($changedState -or $changedVol) {
                        # Prefer the new exact Vizio control
                        if (Get-Command Set-TVVolumeExact -ErrorAction SilentlyContinue) {
                            Write-Host "[TV] Setting volume to $targetVol (exact)" -ForegroundColor Green
                            Set-TVVolumeExact -Target $targetVol
                        }
                        elseif (Get-Command Set-TVVolume -ErrorAction SilentlyContinue) {
                            Write-Host "[TV] Setting volume to $targetVol" -ForegroundColor Green
                            Set-TVVolume -Target $targetVol
                        }
                        else {
                            Write-Host "[TV] No TV volume function found; load tv_vizio_basic.ps1 or tv_setup.ps1 first." -ForegroundColor Red
                        }

                        # Only volume control, no power/apps
                        $lastState  = $stateName
                        $lastVolume = $targetVol
                    } else {
                        Write-Host "[TV] No change in state/volume; leaving volume at $lastVolume" -ForegroundColor DarkGray
                    }
                }
            }
        }
    }
    catch {
        Write-Host "[ERR] Failed to query EDON via HttpClient: $($_.Exception.Message)" -ForegroundColor Red
    }

    Start-Sleep -Seconds $PollSeconds
}
