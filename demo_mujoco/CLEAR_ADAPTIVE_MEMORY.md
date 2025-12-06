# Clear Adaptive Memory for Consistent Performance

## Problem

Adaptive memory may have learned bad patterns from early runs, causing EDON to perform worse. To get consistent green (good performance), you need to clear the adaptive memory database.

## Solution: Clear the Database

### Option 1: Delete the Database File

```bash
# Windows PowerShell
Remove-Item data\robot_stability_memory.db -ErrorAction SilentlyContinue

# Or manually delete:
# data/robot_stability_memory.db
```

### Option 2: Use the API Endpoint

```bash
# Clear memory via API
curl -X POST http://localhost:8000/oem/robot/stability/memory/clear
```

### Option 3: Disable Adaptive Memory

```bash
# Windows PowerShell
$env:EDON_DISABLE_ADAPTIVE_MEMORY="1"
python -m app.main
```

## After Clearing

1. **Restart EDON server** (if using Option 1 or 2)
2. **Run demo** - Adaptive memory will start fresh
3. **First 500 records**: No adjustments (consistent base performance)
4. **After 500 records**: Very small, conservative adjustments

## New Conservative Settings

- **Minimum records**: 500 (was 200)
- **Strategy samples**: 100 per strategy (was 50)
- **Risk baseline**: 200 records (was 100)
- **Adjustments**: 1-2% (was 3-5%)
- **Z-score thresholds**: ±2.0 (was ±1.0 to ±1.5)

This ensures consistent performance for demos!

