# Create GitHub Release for EDON v1.0.1 (PowerShell)

$ErrorActionPreference = "Stop"
$VERSION = "v1.0.1"
$REPO = "edonlabs/edon-cav-engine"  # Update with your org/repo

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Creating GitHub Release $VERSION" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check if gh CLI is installed
try {
    $null = gh --version
} catch {
    Write-Host "ERROR: GitHub CLI (gh) is not installed." -ForegroundColor Red
    Write-Host "Install from: https://cli.github.com/" -ForegroundColor Yellow
    exit 1
}

# Check if authenticated
try {
    gh auth status 2>&1 | Out-Null
} catch {
    Write-Host "ERROR: Not authenticated with GitHub CLI." -ForegroundColor Red
    Write-Host "Run: gh auth login" -ForegroundColor Yellow
    exit 1
}

# Check if files exist
$ZIP_FILE = "EDON_v1.0.1_OEM_RELEASE.zip"
$WHEEL_FILE = "release\${VERSION}\edon-0.1.0-py3-none-any.whl"
$DOCKER_FILE = "release\${VERSION}\edon-server-${VERSION}.docker"
$RELEASE_NOTES = "release\${VERSION}\RELEASE_NOTES.md"

$FILES_TO_UPLOAD = @()

if (Test-Path $ZIP_FILE) {
    Write-Host "[OK] Found: $ZIP_FILE" -ForegroundColor Green
    $FILES_TO_UPLOAD += $ZIP_FILE
} else {
    Write-Host "[WARN] Missing: $ZIP_FILE" -ForegroundColor Yellow
}

if (Test-Path $WHEEL_FILE) {
    Write-Host "[OK] Found: $WHEEL_FILE" -ForegroundColor Green
    $FILES_TO_UPLOAD += $WHEEL_FILE
} elseif (Test-Path "sdk\python\dist\edon-0.1.0-py3-none-any.whl") {
    Write-Host "[OK] Found: sdk\python\dist\edon-0.1.0-py3-none-any.whl" -ForegroundColor Green
    $FILES_TO_UPLOAD += "sdk\python\dist\edon-0.1.0-py3-none-any.whl"
} else {
    Write-Host "[WARN] Missing: $WHEEL_FILE" -ForegroundColor Yellow
}

if (Test-Path $DOCKER_FILE) {
    Write-Host "[OK] Found: $DOCKER_FILE" -ForegroundColor Green
    $FILES_TO_UPLOAD += $DOCKER_FILE
} else {
    Write-Host "[INFO] Missing: $DOCKER_FILE (optional)" -ForegroundColor Cyan
}

if (-not (Test-Path $RELEASE_NOTES)) {
    Write-Host "ERROR: Release notes not found: $RELEASE_NOTES" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Creating release with $($FILES_TO_UPLOAD.Count) asset(s)..." -ForegroundColor Yellow

# Create release
$filesArg = $FILES_TO_UPLOAD -join " "
gh release create "${VERSION}" `
    --title "EDON CAV Engine ${VERSION}" `
    --notes-file "$RELEASE_NOTES" `
    $FILES_TO_UPLOAD

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "[OK] GitHub Release created!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Release URL: https://github.com/${REPO}/releases/tag/${VERSION}" -ForegroundColor Yellow
Write-Host ""
Write-Host "To view: gh release view ${VERSION}" -ForegroundColor Cyan

