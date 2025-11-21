Write-Host "== Starting EDON Humanoid Consumer =="

if (-not $env:EDON_API_BASE) {
    $env:EDON_API_BASE = "http://127.0.0.1:8001"
}

.\.venv\Scripts\python.exe tools\humanoid_consumer.py
