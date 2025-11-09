# How to Start the Server (Correct Way)

## Problem
You're in the wrong directory: `C:\Users\cjbig\Desktop\EDON\data\raw\wesad`
The `app` module is at the project root: `C:\Users\cjbig\Desktop\EDON`

## Solution

### Step 1: Navigate to Project Root
```powershell
cd C:\Users\cjbig\Desktop\EDON
```

### Step 2: Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Start the Server
```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Full Command Sequence

```powershell
# Navigate to project root
cd C:\Users\cjbig\Desktop\EDON

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Verify It's Working

After starting, you should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Then visit: http://localhost:8000/docs

You should see the new memory endpoints!

