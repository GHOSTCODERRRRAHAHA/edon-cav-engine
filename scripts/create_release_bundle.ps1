# Create EDON v1.0.1 OEM Release Bundle (PowerShell)

$ErrorActionPreference = "Stop"
$VERSION = "v1.0.1"
$BUNDLE_NAME = "EDON_${VERSION}_OEM_RELEASE"
$BUNDLE_DIR = $BUNDLE_NAME

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Creating EDON $VERSION OEM Release Bundle" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Clean previous bundle
if (Test-Path $BUNDLE_DIR) {
    Write-Host "Removing previous bundle..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $BUNDLE_DIR
}

if (Test-Path "${BUNDLE_NAME}.zip") {
    Write-Host "Removing previous zip..." -ForegroundColor Yellow
    Remove-Item -Force "${BUNDLE_NAME}.zip"
}

# Create bundle structure
Write-Host "Creating bundle structure..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "${BUNDLE_DIR}\docker" | Out-Null
New-Item -ItemType Directory -Force -Path "${BUNDLE_DIR}\sdk\python" | Out-Null
New-Item -ItemType Directory -Force -Path "${BUNDLE_DIR}\sdk\cpp" | Out-Null
New-Item -ItemType Directory -Force -Path "${BUNDLE_DIR}\docs" | Out-Null

# Copy Docker image (if exists)
$dockerImage = "release\${VERSION}\edon-server-${VERSION}.docker"
$dockerImageGz = "release\${VERSION}\edon-server-${VERSION}.docker.tar.gz"
if (Test-Path $dockerImage) {
    Write-Host "Copying Docker image..." -ForegroundColor Green
    Copy-Item $dockerImage "${BUNDLE_DIR}\docker\edon-server-${VERSION}.tar" -Force
} elseif (Test-Path $dockerImageGz) {
    Write-Host "Copying Docker image (compressed)..." -ForegroundColor Green
    Copy-Item $dockerImageGz "${BUNDLE_DIR}\docker\edon-server-${VERSION}.tar.gz" -Force
} else {
    Write-Host "Warning: Docker image not found. Creating placeholder..." -ForegroundColor Yellow
    @"
Docker image: edon-server:${VERSION}
Build with: docker build -t edon-server:${VERSION} .
Save with: docker save edon-server:${VERSION} > edon-server-${VERSION}.tar
"@ | Out-File -FilePath "${BUNDLE_DIR}\docker\README.txt" -Encoding UTF8
}

# Copy Python SDK wheel
$pythonWheel1 = "release\${VERSION}\edon-0.1.0-py3-none-any.whl"
$pythonWheel2 = "sdk\python\dist\edon-0.1.0-py3-none-any.whl"
if (Test-Path $pythonWheel1) {
    Write-Host "Copying Python SDK wheel..." -ForegroundColor Green
    Copy-Item $pythonWheel1 "${BUNDLE_DIR}\sdk\python\" -Force
} elseif (Test-Path $pythonWheel2) {
    Write-Host "Copying Python SDK wheel from dist..." -ForegroundColor Green
    Copy-Item $pythonWheel2 "${BUNDLE_DIR}\sdk\python\" -Force
} else {
    Write-Host "Warning: Python wheel not found. Creating placeholder..." -ForegroundColor Yellow
    @"
Python SDK wheel: edon-0.1.0-py3-none-any.whl
Build with: cd sdk\python && python -m build --wheel
"@ | Out-File -FilePath "${BUNDLE_DIR}\sdk\python\README.txt" -Encoding UTF8
}

# Copy C++ SDK (if exists)
$cppSdkZip = "release\${VERSION}\edon-cpp-sdk-${VERSION}.zip"
$cppSdkTar = "release\${VERSION}\edon-sdk-cpp-${VERSION}.tar.gz"
if (Test-Path $cppSdkZip) {
    Write-Host "Copying C++ SDK..." -ForegroundColor Green
    Copy-Item $cppSdkZip "${BUNDLE_DIR}\sdk\cpp\" -Force
} elseif (Test-Path $cppSdkTar) {
    Write-Host "Copying C++ SDK (tar.gz)..." -ForegroundColor Green
    Copy-Item $cppSdkTar "${BUNDLE_DIR}\sdk\cpp\" -Force
} else {
    Write-Host "Warning: C++ SDK not found. Creating placeholder..." -ForegroundColor Yellow
    @"
C++ SDK: edon-cpp-sdk-${VERSION}.zip
Build with: cd sdk\cpp && mkdir build && cd build && cmake .. && cmake --build . --config Release
"@ | Out-File -FilePath "${BUNDLE_DIR}\sdk\cpp\README.txt" -Encoding UTF8
}

# Copy documentation
Write-Host "Copying documentation..." -ForegroundColor Yellow
if (Test-Path "docs\OEM_ONBOARDING.md") {
    Copy-Item "docs\OEM_ONBOARDING.md" "${BUNDLE_DIR}\docs\" -Force
} else {
    Write-Host "Warning: OEM_ONBOARDING.md not found" -ForegroundColor Yellow
}

if (Test-Path "docs\OEM_API_CONTRACT.md") {
    Copy-Item "docs\OEM_API_CONTRACT.md" "${BUNDLE_DIR}\docs\" -Force
} else {
    Write-Host "Warning: OEM_API_CONTRACT.md not found" -ForegroundColor Yellow
}

if (Test-Path "release\${VERSION}\RELEASE_NOTES.md") {
    Copy-Item "release\${VERSION}\RELEASE_NOTES.md" "${BUNDLE_DIR}\docs\" -Force
} else {
    Write-Host "Warning: RELEASE_NOTES.md not found" -ForegroundColor Yellow
}

# Create bundle README
$readmeContent = @"
# EDON CAV Engine ${VERSION} - OEM Release Bundle

**Release Date**: 2025-11-20

## Contents

This bundle contains all artifacts for EDON CAV Engine ${VERSION}:

- **docker/**: Docker image tarball
- **sdk/python/**: Python SDK wheel
- **sdk/cpp/**: C++ SDK archive
- **docs/**: OEM documentation

## Quick Start

1. **Load Docker image**:
   ``docker load < docker/edon-server-${VERSION}.tar``

2. **Install Python SDK**:
   ``pip install sdk/python/edon-0.1.0-py3-none-any.whl[grpc]``

3. **Extract C++ SDK**:
   ``unzip sdk/cpp/edon-cpp-sdk-${VERSION}.zip``

4. **Read documentation**:
   - Start with ``docs/OEM_ONBOARDING.md``
   - See ``docs/OEM_API_CONTRACT.md`` for API details
   - See ``docs/RELEASE_NOTES.md`` for release information

## Support

For integration questions, see the documentation or contact the EDON team.
"@

$readmeContent | Out-File -FilePath "${BUNDLE_DIR}\README.md" -Encoding UTF8

# Create zip archive
Write-Host ""
Write-Host "Creating zip archive..." -ForegroundColor Yellow
Compress-Archive -Path $BUNDLE_DIR -DestinationPath "${BUNDLE_NAME}.zip" -Force

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Release bundle created!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Bundle directory: ${BUNDLE_DIR}\" -ForegroundColor Yellow
Write-Host "Zip archive: ${BUNDLE_NAME}.zip" -ForegroundColor Yellow
Write-Host ""
Write-Host "Contents:" -ForegroundColor Yellow
Get-ChildItem -Recurse $BUNDLE_DIR | Select-Object FullName, Length | Format-Table -AutoSize
Write-Host ""
$bundleSize = (Get-ChildItem -Recurse $BUNDLE_DIR | Measure-Object -Property Length -Sum).Sum / 1MB
$zipSize = (Get-Item "${BUNDLE_NAME}.zip").Length / 1MB
Write-Host "Bundle size: $([math]::Round($bundleSize, 2)) MB" -ForegroundColor Yellow
Write-Host "Zip size: $([math]::Round($zipSize, 2)) MB" -ForegroundColor Yellow

