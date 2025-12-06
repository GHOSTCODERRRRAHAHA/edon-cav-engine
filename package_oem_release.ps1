# ============================================================
# 📦 EDON OEM Release Package Creator
# ============================================================
# Creates a complete OEM package with Docker, SDK, docs, examples
# ============================================================

[CmdletBinding()]
param(
    [string]$Version = "v1.0.1",
    [string]$OutputDir = "dist",
    [switch]$IncludeModels = $false  # Include model files separately (if not in Docker)
)

$ErrorActionPreference = "Stop"

# Configuration
$Date = Get-Date -Format "yyyy-MM-dd"
$VersionSlug = ($Version -replace '\.', '_')
$PkgName = "EDON_${VersionSlug}_OEM_RELEASE"
$ZipPath = Join-Path $OutputDir "$PkgName.zip"
$Staging = "temp_oem_package"

Write-Host "============================================================"
Write-Host "📦 Creating EDON OEM Release Package: $PkgName"
Write-Host "============================================================"

# 1. Create staging directory
if (Test-Path $Staging) {
    Remove-Item $Staging -Recurse -Force
}
New-Item -ItemType Directory -Path $Staging | Out-Null

# Create subdirectories
$subdirs = @("docker", "sdk/python", "docs", "examples", "scripts", "models")
foreach ($dir in $subdirs) {
    $fullPath = Join-Path $Staging $dir
    New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
}

# 2. Copy Docker image
Write-Host "[1/8] Copying Docker image..."
$dockerImage = "release/edon-server-v1.0.1.tar"
if (Test-Path $dockerImage) {
    Copy-Item $dockerImage (Join-Path $Staging "docker/") -Force
    Write-Host "  [OK] Docker image copied"
} else {
    Write-Host "  ⚠️  Docker image not found: $dockerImage"
    Write-Host "     Run: docker save edon-server:v1.0.1 -o release/edon-server-v1.0.1.tar"
}

# 3. Copy Python SDK
Write-Host "[2/8] Copying Python SDK..."
$sdkWheel = "sdk/python/dist/edon-*.whl"
$sdkFiles = Get-ChildItem -Path $sdkWheel -ErrorAction SilentlyContinue
if ($sdkFiles) {
    Copy-Item $sdkFiles[0].FullName (Join-Path $Staging "sdk/python/") -Force
    Write-Host "  [OK] SDK wheel copied: $($sdkFiles[0].Name)"
} else {
    Write-Host "  ⚠️  SDK wheel not found. Build it first:"
    Write-Host "     cd sdk/python; python -m build"
}

# 4. Copy OEM Documentation
Write-Host "[3/8] Copying OEM documentation..."
$oemDocs = @(
    "docs/OEM_ONBOARDING.md",
    "docs/OEM_INTEGRATION.md",
    "docs/OEM_ROBOT_STABILITY.md",
    "docs/OEM_API_CONTRACT.md",
    "docs/OEM_BRIEF.md"
)
foreach ($doc in $oemDocs) {
    if (Test-Path $doc) {
        Copy-Item $doc (Join-Path $Staging "docs/") -Force
        Write-Host "  ✅ $doc"
    } else {
        Write-Host "  ⚠️  Missing: $doc"
    }
}

# 5. Copy Examples
Write-Host "[4/8] Copying examples..."
$examples = @(
    "examples/robot_stability_example.py"
)
foreach ($ex in $examples) {
    if (Test-Path $ex) {
        Copy-Item $ex (Join-Path $Staging "examples/") -Force
        Write-Host "  [OK] $ex"
    }
}
# Copy all examples if directory exists
if (Test-Path "examples") {
    Get-ChildItem -Path "examples" -Filter "*.py" | ForEach-Object {
        Copy-Item $_.FullName (Join-Path $Staging "examples/") -Force
    }
}

# 6. Copy Setup Scripts
Write-Host "[5/8] Copying setup scripts..."
$scripts = @(
    "setup_oem_sdk.ps1",
    "start_edon_core_server.ps1"
)
foreach ($script in $scripts) {
    if (Test-Path $script) {
        Copy-Item $script (Join-Path $Staging "scripts/") -Force
        Write-Host "  [OK] $script"
    }
}
# Create Linux/macOS setup script if missing
$setupSh = Join-Path $Staging "scripts/setup_oem_sdk.sh"
if (-not (Test-Path $setupSh)) {
    $shContent = @'
#!/bin/bash
# EDON OEM SDK Setup Script (Linux/macOS)

echo "Setting up EDON OEM SDK..."

# Load Docker image
if [ -f "docker/edon-server-v1.0.1.tar" ]; then
    docker load < docker/edon-server-v1.0.1.tar
    echo "Docker image loaded"
else
    echo "WARNING: Docker image not found"
fi

# Install Python SDK
if ls sdk/python/edon-*.whl 1> /dev/null 2>&1; then
    pip install sdk/python/edon-*.whl
    echo "Python SDK installed"
else
    echo "WARNING: SDK wheel not found"
fi

# Verify installation
python -c "from edon import EdonClient; print('SDK verified')" || echo "WARNING: SDK verification failed"

# Start server
echo "Starting EDON server..."
docker run -d --name edon-server -p 8002:8000 -p 50052:50051 edon-server:v1.0.1 || echo "WARNING: Server already running or Docker not available"

# Health check
sleep 2
curl http://localhost:8002/health || echo "WARNING: Health check failed"

echo "EDON OEM SDK setup complete!"
'@
    # Use Out-File with UTF8 encoding to preserve bash script format
    $shContent | Out-File -FilePath $setupSh -Encoding utf8 -NoNewline
    Write-Host "  Created setup_oem_sdk.sh"
}

# 7. Copy Models (if needed separately)
if ($IncludeModels) {
    Write-Host "[6/8] Copying model files..."
    $models = @(
        "models/cav_state_v3_2.joblib",
        "models/edon_v8_strategy_memory_features.pt",
        "models/edon_fail_risk_v1_fixed_v2.pt"
    )
    foreach ($model in $models) {
        if (Test-Path $model) {
            Copy-Item $model (Join-Path $Staging "models/") -Force
            Write-Host "  [OK] $model"
        }
    }
} else {
    Write-Host "[6/8] Skipping models (included in Docker image)"
}

# 8. Create Release Notes
Write-Host "[7/8] Creating release notes..."
$releaseNotes = @"
# EDON $Version OEM Release Notes

**Release Date:** $Date  
**Version:** $Version  
**Contact:** Charlie Biggins - charlie@edoncore.com

## What's New

### Robot Stability Control ⭐ NEW
- Added `/oem/robot/stability` endpoint
- 97% intervention reduction (validated)
- Real-time control (<25ms latency)
- Integrated into EDON Core API

### Human State Prediction
- `/oem/cav/batch` endpoint
- Adaptive memory engine
- Control scale recommendations

## API Endpoints

- **POST** `/oem/cav/batch` - Human state prediction
- **POST** `/oem/robot/stability` - Robot stability control ⭐ NEW
- **GET** `/health` - Health check
- **GET** `/telemetry` - Performance metrics
- **GET** `/memory/summary` - Adaptive memory stats

## SDK Methods

- `client.cav()` - Human state prediction
- `client.robot_stability()` - Robot stability control ⭐ NEW
- `client.health()` - Health check

## Installation

See `docs/OEM_ONBOARDING.md` for complete installation instructions.

## Support

**Contact:** Charlie Biggins - charlie@edoncore.com

---

*EDON $Version OEM Release*
"@
Set-Content -Path (Join-Path $Staging "RELEASE_NOTES.md") -Value $releaseNotes
Write-Host "  [OK] Release notes created"

# 9. Create README
Write-Host "[8/8] Creating package README..."
$readme = @"
# EDON $Version OEM Release

**Contact:** Charlie Biggins - charlie@edoncore.com

## Quick Start

1. **Extract this package**
2. **Run setup script:**
   - Windows: `.\scripts\setup_oem_sdk.ps1`
   - Linux/macOS: `./scripts/setup_oem_sdk.sh`
3. **Verify installation:**
   ```bash
   curl http://localhost:8002/health
   ```
4. **See documentation:**
   - `docs/OEM_ONBOARDING.md` - Getting started
   - `docs/OEM_INTEGRATION.md` - Integration guide
   - `docs/OEM_ROBOT_STABILITY.md` - Robot stability API

## What's Included

- **Docker Image** - Pre-built EDON Core server
- **Python SDK** - Client library with REST/gRPC support
- **Documentation** - Complete OEM guides
- **Examples** - Working integration code
- **Setup Scripts** - Automated installation

## Capabilities

1. **Human State Prediction** - Adapt robot to operator/occupant state
2. **Robot Stability Control** - Prevent interventions (97% reduction)
3. **Adaptive Memory** - Unsupervised learning and personalization

## Support

**Email:** charlie@edoncore.com

---

*EDON $Version OEM Release - $Date*
"@
Set-Content -Path (Join-Path $Staging "README.md") -Value $readme
Write-Host "  [OK] README created"

# 10. Generate checksums
Write-Host "[9/9] Generating file checksums..."
$checksums = @()
Get-ChildItem -Path $Staging -Recurse -File | ForEach-Object {
    $hash = (Get-FileHash $_.FullName -Algorithm SHA256).Hash
    $relativePath = $_.FullName.Replace((Resolve-Path $Staging).Path + "\", "").Replace("\", "/")
    $checksums += "$hash  $relativePath"
}
Set-Content -Path (Join-Path $Staging "SHA256SUMS.txt") -Value $checksums
Write-Host "  [OK] Checksums generated"

# 11. Create ZIP archive
Write-Host "`n[10/10] Creating ZIP archive..."
if (Test-Path $OutputDir) {
    Remove-Item $OutputDir -Recurse -Force -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

Compress-Archive -Path "$Staging\*" -DestinationPath $ZipPath -Force
Write-Host "  [OK] ZIP archive created: $ZipPath"

# 12. Cleanup
Write-Host "`n[11/11] Cleaning up..."
Remove-Item $Staging -Recurse -Force
Write-Host "  [OK] Cleanup complete"

# Summary
Write-Host "`n============================================================"
Write-Host "[SUCCESS] OEM Package Created Successfully!"
Write-Host "============================================================"
Write-Host "Package: $ZipPath"
Write-Host "Size: $((Get-Item $ZipPath).Length / 1MB) MB"
Write-Host "`nReady to ship to OEMs!"
Write-Host "============================================================"

