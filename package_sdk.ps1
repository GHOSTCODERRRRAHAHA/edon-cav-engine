# ============================================================
# 📦 EDON CAV Engine v3.2 — OEM SDK Packaging with Hash Logging (PS5-compatible)
# ============================================================

[CmdletBinding()]
param(
  [string]$Version = "v3.2",
  [switch]$IncludeParquet,          # add outputs\oem_100k_windows.parquet if present
  [string]$Label = ""               # optional extra label in the zip name
)

$ErrorActionPreference = "Stop"

# 1) Config
$Date        = Get-Date -Format "yyyy-MM-dd"
$VersionSlug = ($Version -replace '\.', '_')
$OutputDir   = "dist"
$PkgName     = "EDON_CAV_${VersionSlug}_OEM_SDK_${Date}"
if ($Label -ne "") { $PkgName = $PkgName + "_" + $Label }
$ZipPath     = Join-Path $OutputDir ($PkgName + ".zip")
$Staging     = "temp_sdk"
$DocsSrc     = "docs"
$Models      = @(
  "models\cav_state_v3_2.joblib",
  "models\cav_state_scaler_v3_2.joblib",
  "models\cav_state_schema_v3_2.json"
)
$Tools       = @(
  "tools\cav_dashboard.py",
  "tools\add_v32_features.py"
)
$DemoCandidates = @("demo\demo_infer_example.py","demo_infer.py")
$ReqFile     = "requirements.txt"
$Parquet     = "outputs\oem_100k_windows.parquet"

# 2) Prep folders
if (-not (Test-Path $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir | Out-Null }
if (Test-Path $Staging) { Remove-Item $Staging -Recurse -Force }
New-Item -ItemType Directory -Path $Staging | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Staging "docs") | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Staging "models") | Out-Null

# 3) Copy models (required) into models/ subdirectory
foreach ($m in $Models) {
  if (-not (Test-Path $m)) { throw "Missing required model artifact: $m" }
  $fileName = Split-Path -Path $m -Leaf
  $destPath = Join-Path -Path $Staging -ChildPath "models" | Join-Path -ChildPath $fileName
  Copy-Item $m $destPath -Force
}

# 4) Copy tools (required)
foreach ($t in $Tools) {
  if (-not (Test-Path $t)) { throw "Missing required tool: $t" }
  Copy-Item $t $Staging -Force
}

# 5) Copy demo (one of the candidates must exist)
$DemoPicked = $null
foreach ($d in $DemoCandidates) {
  if (Test-Path $d) { $DemoPicked = $d; break }
}
if (-not $DemoPicked) { throw "Missing demo script (demo\demo_infer_example.py or demo_infer.py)" }
Copy-Item $DemoPicked (Join-Path $Staging "demo_infer.py") -Force

# 6) Copy requirements
if (-not (Test-Path $ReqFile)) { throw "Missing requirements.txt" }
Copy-Item $ReqFile $Staging -Force

# 7) Copy docs
if (-not (Test-Path $DocsSrc)) { throw "docs\ folder not found. Move your *.md docs into docs\ first." }
Copy-Item "$DocsSrc\*" (Join-Path $Staging "docs") -Recurse -Force

# 8) Optionally include parquet
if ($IncludeParquet -and (Test-Path $Parquet)) {
  Copy-Item $Parquet (Join-Path $Staging "oem_100k_windows.parquet") -Force
}

# 8.5) Copy helper scripts if they exist
$HelperScripts = @("run_demo.ps1", "verify_hashes.ps1")
foreach ($script in $HelperScripts) {
  if (Test-Path $script) {
    Copy-Item $script (Join-Path $Staging $script) -Force
  }
}

# 9) Compute SHA256 hashes for provenance (models + staged files)
function Get-HashLine([string]$p) {
  $h = Get-FileHash $p -Algorithm SHA256
  # Create a path relative to staging if possible
  $rel = try {
    $full = [IO.Path]::GetFullPath($p)
    $stg  = [IO.Path]::GetFullPath($Staging)
    if ($full.StartsWith($stg)) { $full.Substring($stg.Length).TrimStart('\') } else { $p }
  } catch { $p }
  return @{ Path = $rel; Hash = $h.Hash }
}

$HashList = @()
# Hash required model artifacts (from project root)
foreach ($m in $Models) { $HashList += Get-HashLine (Join-Path $PWD $m) }
# Hash everything that will actually ship (from staging)
Get-ChildItem -Path $Staging -Recurse -File | ForEach-Object {
  $HashList += Get-HashLine $_.FullName
}

# 10) Try to get git tag/commit (best-effort; PS5-safe)
$GitTag = ""
$GitCommit = ""
try { $GitTag    = (git describe --tags 2>$null) } catch { }
try { $GitCommit = (git rev-parse --short HEAD 2>$null) } catch { }
if ([string]::IsNullOrEmpty($GitTag))   { $GitTag = "n/a" }
if ([string]::IsNullOrEmpty($GitCommit)){ $GitCommit = "n/a" }

# 11) Write README_SDK.txt (includes quick start + provenance note)
$Readme = @"
# EDON CAV Engine OEM SDK

Version: $Version
Date: $Date
Git Tag: $GitTag
Commit:  $GitCommit

## Contents
- Models: LightGBM v3.2
- Tools: cav_dashboard.py, add_v32_features.py
- Demo: demo_infer.py
- Docs: OEM brief, license, NDA, and command guides

## Quick Start
pip install -r requirements.txt
python demo_infer.py
streamlit run cav_dashboard.py

## Notes
If included, oem_100k_windows.parquet contains pre-built windows for OEM evaluation.
Verify integrity with HASHES.txt (SHA256).
Contact: oem@edon.ai for license & OEM support.
"@

Set-Content -Path (Join-Path $Staging "README_SDK.txt") -Value $Readme -Encoding UTF8
# 12) Write HASHES.txt and MANIFEST.txt
function Get-HashLine([string]$p) {
    $h = Get-FileHash $p -Algorithm SHA256
    $rel = try {
        $full = [IO.Path]::GetFullPath($p)
        $stg  = [IO.Path]::GetFullPath($Staging)
        if ($full.StartsWith($stg)) { $full.Substring($stg.Length).TrimStart('\') } else { $p }
    } catch { $p }
    return "{0}  {1}" -f $h.Hash, $rel
}

$HashLines = @()
Get-ChildItem -Path $Staging -Recurse -File | ForEach-Object {
    $HashLines += Get-HashLine $_.FullName
}

Set-Content -Path (Join-Path $Staging "HASHES.txt") -Value $HashLines -Encoding ASCII

$Manifest = (Get-ChildItem -Path $Staging -Recurse -File) |
    Sort-Object FullName |
    ForEach-Object { "{0}`t{1} bytes" -f ($_.FullName.Substring([IO.Path]::GetFullPath($Staging).Length+1)), $_.Length }
Set-Content -Path (Join-Path $Staging "MANIFEST.txt") -Value $Manifest -Encoding ASCII
# 13) Zip package
if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
Compress-Archive -Path (Join-Path $Staging '*') -DestinationPath $ZipPath

# 14) Success summary
$zipInfo = Get-Item $ZipPath
Write-Host ""
Write-Host "✅ SDK packaged" -ForegroundColor Green
Write-Host "  Path : $($zipInfo.FullName)"
Write-Host "  Size : $([Math]::Round($zipInfo.Length/1MB,2)) MB"
Write-Host "  Files: $((Get-ChildItem -Path $Staging -Recurse -File).Count)"
