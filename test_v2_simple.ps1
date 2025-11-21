# Simple v2 API test script

$baseUrl = "http://127.0.0.1:8001"

Write-Host "Testing EDON v2 API" -ForegroundColor Green
Write-Host "Base URL: $baseUrl"
Write-Host ""

# Check health first
Write-Host "1. Checking health..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "$baseUrl/health" -Method GET
    Write-Host "   Mode: $($health.mode)" -ForegroundColor $(if ($health.mode -eq "v2") { "Green" } else { "Red" })
    Write-Host "   Engine: $($health.engine)"
    
    if ($health.mode -ne "v2") {
        Write-Host ""
        Write-Host "⚠ Server is not in v2 mode!" -ForegroundColor Yellow
        Write-Host "   Restart server with: `$env:EDON_MODE='v2'" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "   ✗ Health check failed: $_" -ForegroundColor Red
    exit 1
}

# Test v2 batch endpoint
Write-Host ""
Write-Host "2. Testing v2 batch endpoint..." -ForegroundColor Yellow

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

try {
    $response = Invoke-RestMethod -Uri "$baseUrl/v2/oem/cav/batch" `
        -Method POST `
        -Body $body `
        -ContentType "application/json"
    
    $result = $response.results[0]
    
    Write-Host "   ✓ Success!" -ForegroundColor Green
    Write-Host "   State: $($result.state_class)" -ForegroundColor Cyan
    Write-Host "   Stress: $($result.p_stress)" -ForegroundColor Cyan
    Write-Host "   Chaos: $($result.p_chaos)" -ForegroundColor Cyan
    Write-Host "   CAV Vector: $($result.cav_vector.Count) dimensions"
    Write-Host "   Latency: $($response.latency_ms) ms"
    
} catch {
    Write-Host "   ✗ Failed: $_" -ForegroundColor Red
    if ($_.Exception.Response.StatusCode -eq 404) {
        Write-Host "   Endpoint not found. Make sure:" -ForegroundColor Yellow
        Write-Host "   1. Server is running with EDON_MODE=v2" -ForegroundColor Yellow
        Write-Host "   2. Using correct endpoint: /v2/oem/cav/batch" -ForegroundColor Yellow
    }
    exit 1
}

Write-Host ""
Write-Host "✓ All tests passed!" -ForegroundColor Green

