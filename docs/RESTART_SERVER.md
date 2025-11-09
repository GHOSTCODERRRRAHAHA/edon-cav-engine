# How to Restart the FastAPI Server

## What Server Needs Restarting?

The **EDON CAV API FastAPI server** running on port 8000 needs to be restarted to load the new Adaptive Memory Engine routes.

## Current Status

The server is currently running an older version that doesn't have the memory engine endpoints.

## How to Restart

### Option 1: If server is running in a terminal window

1. **Find the terminal window** where the server is running
2. **Press `Ctrl+C`** to stop the server
3. **Restart it** with:
   ```powershell
   .\venv\Scripts\Activate.ps1
   uvicorn app.main:app --reload
   ```

### Option 2: If server is running in background

1. **Find and kill the process**:
   ```powershell
   # Find Python processes
   tasklist | findstr python
   
   # Kill the process (replace PID with actual process ID)
   taskkill /PID <PID> /F
   ```

2. **Start the server**:
   ```powershell
   .\venv\Scripts\Activate.ps1
   uvicorn app.main:app --reload
   ```

### Option 3: Start fresh

1. **Navigate to project root**:
   ```powershell
   cd C:\Users\cjbig\Desktop\EDON
   ```

2. **Activate virtual environment**:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

3. **Start the server**:
   ```powershell
   uvicorn app.main:app --reload
   ```

The `--reload` flag will automatically restart the server when code changes are detected.

## Verify Server is Running

After restarting, verify the new endpoints are available:

```powershell
# Check root endpoint
curl http://localhost:8000/

# Should show new endpoints including:
# - "memory_summary": "GET /memory/summary"
# - "memory_clear": "POST /memory/clear"
```

Or visit: http://localhost:8000/docs to see the API documentation with the new memory endpoints.

## What Will Be Available After Restart

✅ `POST /cav` - Now includes `adaptive` field in response
✅ `GET /memory/summary` - Get 24-hour memory statistics
✅ `POST /memory/clear` - Clear memory (for testing)

