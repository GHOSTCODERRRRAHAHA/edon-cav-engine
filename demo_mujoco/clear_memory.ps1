# PowerShell script to clear adaptive memory

Write-Host "Clearing EDON Adaptive Memory..." -ForegroundColor Yellow

# Option 1: Clear via API
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/oem/robot/stability/memory/clear" -Method POST -ErrorAction Stop
    Write-Host "✓ Memory cleared via API" -ForegroundColor Green
    Write-Host $response.Content
} catch {
    Write-Host "✗ Could not clear via API (server may not be running)" -ForegroundColor Red
    Write-Host "  Trying to delete database file directly..." -ForegroundColor Yellow
    
    # Option 2: Delete database file
    $dbPath = "data\robot_stability_memory.db"
    if (Test-Path $dbPath) {
        Remove-Item $dbPath -Force
        Write-Host "✓ Database file deleted: $dbPath" -ForegroundColor Green
    } else {
        Write-Host "✗ Database file not found: $dbPath" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Restart EDON server (if it's running)" -ForegroundColor White
Write-Host "2. Run demo - adaptive memory will start fresh" -ForegroundColor White
Write-Host "3. First 500 records: No adjustments (consistent base performance)" -ForegroundColor White

