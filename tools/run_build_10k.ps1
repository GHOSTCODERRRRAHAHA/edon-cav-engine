Write-Host "== Building EDON 10K OEM Dataset =="

if (-not $env:EDON_API_BASE) {
    $env:EDON_API_BASE = "http://127.0.0.1:8001"
}

.\.venv\Scripts\python.exe tools\build_oem_dataset_10k.py
