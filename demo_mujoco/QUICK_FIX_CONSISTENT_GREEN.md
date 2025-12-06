# Quick Fix: Get Consistent Green Performance

## Problem
You're seeing variable performance (-50%, +50%, -150%) instead of consistent green.

## Root Cause
Adaptive memory is learning from noisy early data and making bad adjustments.

## Solution: Disable Adaptive Memory

### Step 1: Stop Your EDON Server
Press `Ctrl+C` in the terminal running `python -m app.main`

### Step 2: Restart with Adaptive Memory Disabled
```powershell
$env:EDON_DISABLE_ADAPTIVE_MEMORY = "1"
python -m app.main
```

### Step 3: Run Demo
```powershell
cd demo_mujoco
python run_demo.py
```

## Or Use the Start Script (Easier)
```powershell
cd demo_mujoco
.\start_demo.ps1
```

This automatically disables adaptive memory.

## Verify It's Working

Check EDON server logs. You should see:
```
[ROBOT-STABILITY] Adaptive memory DISABLED (EDON_DISABLE_ADAPTIVE_MEMORY=1) - using base v8 policy only
```

## Expected Results

With adaptive memory **disabled**:
- ✅ Consistent base v8 policy performance
- ✅ No learning from noisy data
- ✅ Predictable zero-shot results (typically 25-50% improvement)

With adaptive memory **enabled**:
- ⚠️ Variable performance (can be worse or better)
- ⚠️ Learns from early noisy data
- ⚠️ Needs 500+ records before reliable

## For OEM Demos

**Always disable adaptive memory** for consistent, predictable results.

