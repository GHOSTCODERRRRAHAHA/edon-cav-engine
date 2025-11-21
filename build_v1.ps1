# build_v1.ps1
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

Write-Host "`n=== EDON v1.0 Build: parse -> merge -> train -> reload ===`n" -ForegroundColor Cyan

$py = ".\venv\Scripts\python.exe"
$dataRoot = "data\unified"
$modelOut = "models\cav_engine_v4_0"

if (-not (Test-Path $py)) { throw "Python venv missing at $py" }
New-Item -ItemType Directory -Force -Path $dataRoot | Out-Null

# --- 1) Parse external datasets (comment these 2 lines out if you don't have raw files yet) ---
try {
  & $py tools\parse_wisdm.py   --src data\external\wisdm   --out "$dataRoot\wisdm.jsonl"
} catch { Write-Host "parse_wisdm.py skipped or produced 0 windows" -ForegroundColor DarkYellow }
try {
  & $py tools\parse_mobiact.py --src data\external\mobiact --out "$dataRoot\mobiact.jsonl"
} catch { Write-Host "parse_mobiact.py skipped or produced 0 windows" -ForegroundColor DarkYellow }

# --- 2) Merge or use prebuilt ---
$wisdm = Join-Path $dataRoot "wisdm.jsonl"
$mobia = Join-Path $dataRoot "mobiact.jsonl"
$all   = Join-Path $dataRoot "all_v10.jsonl"

if ((Test-Path $wisdm) -and (Test-Path $mobia)) {
  Get-Content $wisdm, $mobia | Set-Content $all
} elseif (-not (Test-Path $all)) {
  throw "No training data found. Provide $all or ensure both $wisdm and $mobia exist."
}

# --- 3) Train ---
& $py tools\train_cav_model.py --data $all --out $modelOut --pca 128

# --- 4) Restart API ---
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object {
  Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
}
Start-Process powershell -ArgumentList "-NoProfile","-Command","$py -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"
Start-Sleep -Seconds 5

# --- 5) Verify ---
Write-Host "`nChecking /models/info" -ForegroundColor Yellow
curl http://127.0.0.1:8000/models/info
Write-Host "`nChecking /health" -ForegroundColor Yellow
curl http://127.0.0.1:8000/health

Write-Host ""
Write-Host "[OK] Build complete - EDON v1.0 SDK ready." -ForegroundColor Green
