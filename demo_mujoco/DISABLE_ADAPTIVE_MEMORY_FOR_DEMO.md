# Disable Adaptive Memory for Consistent Demo Performance

## Problem
Adaptive memory can cause variability in zero-shot performance, leading to inconsistent results (sometimes -50%, sometimes +50%).

## Solution
**Disable adaptive memory for demos** to get consistent base v8 policy performance.

## How to Disable

### Option 1: Use the Start Script (Recommended)
```powershell
.\start_demo.ps1
```
This automatically disables adaptive memory.

### Option 2: Set Environment Variable Before Starting Demo
```powershell
$env:EDON_DISABLE_ADAPTIVE_MEMORY = "1"
python run_demo.py
```

### Option 3: Set Environment Variable Before Starting EDON Server
```powershell
# In the terminal where you run the EDON server:
$env:EDON_DISABLE_ADAPTIVE_MEMORY = "1"
python -m app.main
```

## Verify It's Disabled

Check the EDON server logs. You should see:
```
[ROBOT-STABILITY] Adaptive memory disabled (EDON_DISABLE_ADAPTIVE_MEMORY=1)
```

If you see:
```
[ROBOT-STABILITY] Adaptive memory applied: ...
```
Then adaptive memory is still enabled.

## Why Disable for Demos?

1. **Consistency**: Base v8 policy gives consistent zero-shot performance
2. **Predictability**: No learning from noisy early data
3. **Fair Comparison**: Same policy for every run

## When to Enable

Enable adaptive memory when:
- You want EDON to learn and improve over time
- You have many episodes (500+ records)
- You're doing long-term testing, not demos

## Current Status

The demo now **disables adaptive memory by default** for consistent performance.

