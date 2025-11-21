# Publish EDON Python SDK to PyPI (PowerShell)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Publishing EDON Python SDK to PyPI" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check if in correct directory
if (-not (Test-Path "sdk\python\pyproject.toml")) {
    Write-Host "ERROR: Must run from repo root" -ForegroundColor Red
    exit 1
}

Push-Location sdk\python

# Check if build tools are installed
try {
    python -c "import build" 2>&1 | Out-Null
} catch {
    Write-Host "Installing build tools..." -ForegroundColor Yellow
    pip install build twine
}

# Clean previous builds
Write-Host "Cleaning previous builds..." -ForegroundColor Yellow
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue

# Build wheel and source distribution
Write-Host "Building wheel and source distribution..." -ForegroundColor Yellow
python -m build

# Check what was built
Write-Host ""
Write-Host "Built artifacts:" -ForegroundColor Green
Get-ChildItem dist\ | Format-Table Name, @{Label="Size (KB)"; Expression={[math]::Round($_.Length/1KB, 2)}} -AutoSize

# Verify with twine
Write-Host ""
Write-Host "Verifying package..." -ForegroundColor Yellow
twine check dist\*

# Ask for confirmation
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Ready to upload to PyPI" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Files to upload:" -ForegroundColor Yellow
Get-ChildItem dist\ | Format-Table Name, Length -AutoSize
Write-Host ""
$confirm = Read-Host "Upload to PyPI? (yes/no)"

if ($confirm -ne "yes") {
    Write-Host "Upload cancelled." -ForegroundColor Yellow
    Pop-Location
    exit 0
}

# Upload to PyPI
Write-Host ""
Write-Host "Uploading to PyPI..." -ForegroundColor Yellow
twine upload dist\*

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "[OK] Published to PyPI!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Install with: pip install edon" -ForegroundColor Yellow
Write-Host "Or with gRPC: pip install edon[grpc]" -ForegroundColor Yellow

Pop-Location

