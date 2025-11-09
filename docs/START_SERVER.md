# How to Start the Correct Server

## Current Situation

You're likely running the **old server** (`data/raw/wesad/cav_api.py`) which doesn't have the memory engine.

You need to run the **new server** (`app/main.py`) which has the Adaptive Memory Engine.

## How to Start the New Server

### Step 1: Stop the Old Server
If you have a terminal running the old server:
- Press `Ctrl+C` to stop it

### Step 2: Start the New Server

**Option A: Using uvicorn (recommended)**
```powershell
cd C:\Users\cjbig\Desktop\EDON
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Option B: Running directly**
```powershell
cd C:\Users\cjbig\Desktop\EDON
.\venv\Scripts\Activate.ps1
python app/main.py
```

## Verify It's the Right Server

After starting, check:
```powershell
curl http://localhost:8000/
```

You should see:
```json
{
  "service": "EDON CAV Engine",
  "version": "0.1.0",
  "endpoints": {
    "single": "POST /cav",
    "batch": "POST /oem/cav/batch",
    "health": "GET /health",
    "telemetry": "GET /telemetry",
    "memory_summary": "GET /memory/summary",  ← NEW!
    "memory_clear": "POST /memory/clear",     ← NEW!
    "docs": "/docs"
  }
}
```

If you see `"memory_summary"` and `"memory_clear"` in the endpoints, you're running the correct server!

## Quick Test

Visit: http://localhost:8000/docs

You should see:
- ✅ `POST /cav` (with `adaptive` field in response)
- ✅ `GET /memory/summary`
- ✅ `POST /memory/clear`

