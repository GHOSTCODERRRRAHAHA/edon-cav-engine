# Set API Keys as Environment Variables
# Run this before starting the API server

Write-Host "`n=== Setting API Keys ===`n" -ForegroundColor Cyan

# OpenWeatherMap API Key
$env:OPENWEATHER_API_KEY = "23ae82c2010dc3d219600937a66250f6"
Write-Host "✓ OPENWEATHER_API_KEY set" -ForegroundColor Green

# AirNow API Key
$env:AIRNOW_API_KEY = "DDEF6694-BDCC-4ED4-8EC0-F929ABA11DEA"
Write-Host "✓ AIRNOW_API_KEY set" -ForegroundColor Green

# Optional: EDON API Token (for authentication)
# $env:EDON_API_TOKEN = "your-secret-token"
# $env:EDON_AUTH_ENABLED = "true"

Write-Host "`nAPI keys are set for this PowerShell session." -ForegroundColor Yellow
Write-Host "To make them permanent, add to your PowerShell profile or use .env file.`n" -ForegroundColor Gray

