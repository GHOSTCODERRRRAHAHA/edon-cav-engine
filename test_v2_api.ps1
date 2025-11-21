# PowerShell script to test v2 API

Write-Host "=== EDON v2 API Test ===" -ForegroundColor Green
Write-Host ""

# Create test window with correct v2 format
$window = @{
    physio = @{
        EDA = @(0.25) * 240
        BVP = @(0.5) * 240
    }
    motion = @{
        ACC_x = @(0.0) * 240
        ACC_y = @(0.0) * 240
        ACC_z = @(1.0) * 240
    }
    env = @{
        temp_c = 22.0
        humidity = 45.0
        aqi = 20
    }
    task = @{
        id = "test"
        complexity = 0.5
    }
}

$body = @{
    windows = @($window)
} | ConvertTo-Json -Depth 10

Write-Host "Endpoint: http://127.0.0.1:8001/v2/oem/cav/batch" -ForegroundColor Cyan
Write-Host ""

try {
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:8001/v2/oem/cav/batch" `
        -Method POST `
        -Body $body `
        -ContentType "application/json"
    
    Write-Host "✓ Success!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Result:" -ForegroundColor Yellow
    Write-Host "  State: $($response.results[0].state_class)"
    Write-Host "  Stress: $($response.results[0].p_stress)"
    Write-Host "  Chaos: $($response.results[0].p_chaos)"
    Write-Host "  CAV Vector Length: $($response.results[0].cav_vector.Count)"
    Write-Host "  Latency: $($response.latency_ms) ms"
    Write-Host ""
    Write-Host "Influences:" -ForegroundColor Yellow
    Write-Host "  Speed Scale: $($response.results[0].influences.speed_scale)"
    Write-Host "  Safety Scale: $($response.results[0].influences.safety_scale)"
    Write-Host "  Caution Flag: $($response.results[0].influences.caution_flag)"
    
} catch {
    Write-Host "✗ Error: $_" -ForegroundColor Red
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "Response: $responseBody" -ForegroundColor Red
    }
}

