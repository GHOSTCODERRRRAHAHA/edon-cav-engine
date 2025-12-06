# Quick Start: Test EDON Core with Robot Control

## Step 1: Activate Virtual Environment

```powershell
# Activate venv
.\.venv\Scripts\Activate.ps1
```

## Step 2: Start EDON Core Server

```powershell
# Option A: Use the script (auto-activates venv)
.\start_edon_core_server.ps1

# Option B: Manual start
python -m uvicorn app.main:app --host 127.0.0.1 --port 8002
```

**Keep this terminal open** - the server runs in the foreground.

## Step 3: Verify Server is Running

**In a NEW terminal** (keep server running in first terminal):

```powershell
# Activate venv in new terminal
.\.venv\Scripts\Activate.ps1

# Health check
curl http://127.0.0.1:8002/health

# Or test with Python
python -c "import requests; print(requests.get('http://127.0.0.1:8002/health').json())"
```

**Expected output:**
```json
{"ok": true, "model": "...", "uptime_s": ...}
```

## Step 4: Run the Test

**In the same terminal as Step 3:**

```powershell
# Make sure venv is activated
.\.venv\Scripts\Activate.ps1

# Run the test
python test_edon_core_with_robot_control.py
```

## What the Test Does

1. **Test 1: v8 Only (Baseline)**
   - Runs robot episodes with v8 strategy policy
   - Measures: interventions, stability, episode length
   - **Expected**: 1.00 interventions/episode

2. **Test 2: v8 + EDON Core Control Scales**
   - Runs robot episodes with v8 strategy policy
   - **Also** uses EDON Core control scales to modulate actions
   - Measures: interventions, stability, episode length
   - **Question**: Does EDON Core help or hurt?

3. **Comparison**
   - Compares results between v8 only vs v8 + EDON Core
   - Shows if EDON Core provides additional benefit

## Expected Results

### If EDON Core Helps:
- **v8 Only**: 1.00 interventions/episode
- **v8 + EDON Core**: 0.5-0.8 interventions/episode
- **Conclusion**: EDON Core's control scales provide additional safety

### If EDON Core Has No Effect:
- **v8 Only**: 1.00 interventions/episode
- **v8 + EDON Core**: 1.00 interventions/episode
- **Conclusion**: v8 is already optimal, EDON Core doesn't add value for robot stability

### If EDON Core Hurts:
- **v8 Only**: 1.00 interventions/episode
- **v8 + EDON Core**: 1.5-2.0 interventions/episode
- **Conclusion**: EDON Core's control scales conflict with v8's modulations

---

## Troubleshooting

### Server Won't Start

**Missing dependencies:**
```powershell
.\.venv\Scripts\Activate.ps1
pip install dash plotly
```

**Port already in use:**
```powershell
# Use different port
python -m uvicorn app.main:app --host 127.0.0.1 --port 8003
# Then update test script to use port 8003
```

### Test Script Fails

**EDON Core SDK not installed:**
```powershell
.\.venv\Scripts\Activate.ps1
pip install -e sdk/python
```

**v8 model not found:**
```powershell
# Check if model exists
ls models/edon_v8_strategy_memory_features.pt

# If missing, train it first:
python training/train_edon_v8_strategy.py --episodes 300
```

---

*Test script: `test_edon_core_with_robot_control.py`*

