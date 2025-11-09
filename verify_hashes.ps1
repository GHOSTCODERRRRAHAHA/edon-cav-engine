# ============================================================
# EDON CAV Engine v3.2 - Hash Verification Script
# ============================================================
# Verifies SHA256 hashes from HASHES.txt file
#

[CmdletBinding()]
param(
    [string]$HashesFile = "HASHES.txt",  # Path to HASHES.txt
    [switch]$ShowAll                    # Show all files, not just mismatches
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "EDON CAV Engine v3.2 - Hash Verification" -ForegroundColor Cyan
Write-Host ""

# Check if HASHES.txt exists
if (-not (Test-Path $HashesFile)) {
    Write-Host "[ERROR] HASHES.txt not found: $HashesFile" -ForegroundColor Red
    Write-Host "   Run this script from the SDK root directory." -ForegroundColor Yellow
    exit 1
}

Write-Host "Reading hashes from: $HashesFile" -ForegroundColor Green
Write-Host ""

# Read hash file
$hashLines = Get-Content $HashesFile
$totalFiles = 0
$verifiedFiles = 0
$failedFiles = 0
$missingFiles = 0

foreach ($line in $hashLines) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    
    # Parse hash line: "HASH  path/to/file"
    $parts = $line -split '\s+', 2
    if ($parts.Length -ne 2) { continue }
    
    $expectedHash = $parts[0].Trim()
    $filePath = $parts[1].Trim()
    
    $totalFiles++
    
    # Check if file exists
    if (-not (Test-Path $filePath)) {
        Write-Host "[MISSING] $filePath" -ForegroundColor Red
        $missingFiles++
        continue
    }
    
    # Compute actual hash
    $actualHash = (Get-FileHash $filePath -Algorithm SHA256).Hash
    
    # Compare
    if ($actualHash -eq $expectedHash) {
        $verifiedFiles++
        if ($ShowAll) {
            Write-Host "[OK] $filePath" -ForegroundColor Green
        }
    } else {
        $failedFiles++
        Write-Host "[MISMATCH] $filePath" -ForegroundColor Red
        Write-Host "   Expected: $expectedHash" -ForegroundColor Yellow
        Write-Host "   Actual:   $actualHash" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "  Total files:    $totalFiles" -ForegroundColor White
Write-Host "  Verified:      $verifiedFiles" -ForegroundColor Green
Write-Host "  Failed:        $failedFiles" -ForegroundColor $(if ($failedFiles -eq 0) { "Green" } else { "Red" })
Write-Host "  Missing:       $missingFiles" -ForegroundColor $(if ($missingFiles -eq 0) { "Green" } else { "Red" })
Write-Host ""

if ($failedFiles -eq 0 -and $missingFiles -eq 0) {
    Write-Host "All files verified successfully!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "Verification failed. Some files are missing or corrupted." -ForegroundColor Red
    exit 1
}

