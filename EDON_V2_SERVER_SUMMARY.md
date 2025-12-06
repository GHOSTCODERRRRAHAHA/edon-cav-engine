# EDON v2 Server Configuration Summary

## Real EDON v2 Server Implementation

**File**: `app/main.py`

This is the FastAPI application that serves the EDON v2 API. When `EDON_MODE=v2` environment variable is set, it:
- Initializes the v2 engine (`CAVEngineV2`)
- Loads PCA and neural head models (if available)
- Registers v2 routes including `/v2/oem/cav/batch`
- Serves on the configured port (default 8001)

**Entry Point**: `python -m uvicorn app.main:app --host 0.0.0.0 --port 8001`

**Health Endpoint**: `GET /health` (returns `{"mode": "v2", ...}` when in v2 mode)

**API Endpoint**: `POST /v2/oem/cav/batch`

---

## How to Start Manually

### Option 1: Direct Python (Recommended)

```bash
# Set environment variables
export EDON_MODE=v2
export EDON_PCA_PATH=models/pca.pkl          # Optional
export EDON_NEURAL_WEIGHTS=models/neural_head.pt  # Optional

# Start server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

**Windows PowerShell:**
```powershell
$env:EDON_MODE = "v2"
$env:EDON_PCA_PATH = ".\models\pca.pkl"
$env:EDON_NEURAL_WEIGHTS = ".\models\neural_head.pt"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

### Option 2: Using Existing Script

**Windows:**
```powershell
.\start_edon_v2.ps1
```
Note: This script uses port 8002, not 8001.

### Option 3: Docker

```bash
# Build and run
docker-compose up --build

# Or manually:
docker run -p 8001:8001 \
  -e EDON_MODE=v2 \
  -e EDON_PCA_PATH=/app/models/pca.pkl \
  -e EDON_NEURAL_WEIGHTS=/app/models/neural_head.pt \
  edon-server:v2.0.0 \
  python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

---

## How to Start via `start_edon_v2_server.py`

**Simple:**
```bash
python start_edon_v2_server.py
```

**What it does:**
1. Checks if server is already running (via `/health` endpoint)
2. Verifies `app/main.py` exists
3. Checks for model files (warns if missing but continues)
4. Auto-detects uvicorn installation
5. Sets `EDON_MODE=v2` environment variable
6. Sets model paths if files exist
7. Starts server on `127.0.0.1:8001`
8. Waits for server to respond
9. Streams server output

**Features:**
- Auto-detection of server status
- Model file detection
- Graceful error handling
- Real-time output streaming
- Ctrl+C to stop

---

## How `run_eval.py` Behaves When EDON is Offline

### When EDON Server is Not Running:

1. **Health Check Phase:**
   - Attempts to connect to `http://127.0.0.1:8001/health`
   - If connection fails, prints:
     ```
     WARNING: EDON v2 server is not running.
       Health check failed: http://127.0.0.1:8001/health
       Run: python start_edon_v2_server.py
       Evaluation will continue but EDON calls will fail.
     ```

2. **Client Initialization:**
   - Still attempts to create `EdonClient` object
   - If initialization fails, prints warning but continues

3. **During Evaluation:**
   - If `--mode edon` is used, EDON calls will fail
   - Each failed call will log an error but episode continues
   - Metrics will be collected but EDON state will be `None`
   - Evaluation completes with warnings, does not crash

4. **Error Handling:**
   - All EDON-related errors are caught and logged as warnings
   - Evaluation continues even if EDON is completely unavailable
   - Final results are saved regardless of EDON status

### When EDON Server is Running but Wrong Mode:

- If server is in v1 mode, prints:
  ```
  WARNING: EDON server is running but not in v2 mode (mode: v1)
    Run: python start_edon_v2_server.py
  ```

### When EDON Server is Running Correctly:

- Prints: `✓ EDON v2 server is running (mode: v2)`
- Evaluation proceeds normally
- All EDON calls succeed

---

## Verification

**Check if server is running:**
```bash
curl http://127.0.0.1:8001/health
```

**Expected response (v2 mode):**
```json
{
  "ok": true,
  "mode": "v2",
  "engine": "v2",
  "neural_loaded": true/false,
  "pca_loaded": true/false,
  "uptime_s": 123.45
}
```

**Test v2 endpoint:**
```bash
curl -X POST http://127.0.0.1:8001/v2/oem/cav/batch \
  -H "Content-Type: application/json" \
  -d '{"windows": [{"physio": {"EDA": [0.1]*240, "BVP": [0.5]*240}}]}'
```

---

## Troubleshooting

**Server won't start:**
- Check uvicorn is installed: `pip install uvicorn`
- Check `app/main.py` exists
- Check Python version (3.10+)

**Server starts but health check fails:**
- Check port 8001 is not in use: `netstat -an | grep 8001`
- Check firewall settings
- Verify `EDON_MODE=v2` is set

**Models not loading:**
- Check `models/pca.pkl` and `models/neural_head.pt` exist
- Server will run without models (uses defaults)
- Check file permissions

**Evaluation fails with EDON errors:**
- Run `python start_edon_v2_server.py` in separate terminal
- Verify server responds: `curl http://127.0.0.1:8001/health`
- Check server logs for errors

