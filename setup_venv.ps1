# Setup script for EDON project virtual environment
# Run this in PowerShell (may need to run as Administrator)

Write-Host "Setting up virtual environment for EDON project..." -ForegroundColor Green

# Navigate to project directory
cd C:\Users\cjbig\Desktop\EDON

# Remove old venv if it exists
if (Test-Path "venv") {
    Write-Host "Removing old virtual environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force venv
}

# Create new virtual environment
Write-Host "Creating new virtual environment..." -ForegroundColor Green
python -m venv venv

# Activate and install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Green
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "`nVirtual environment setup complete!" -ForegroundColor Green
Write-Host "To activate in the future, run: .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan

