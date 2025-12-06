# How to Disable Adaptive Memory

## Quick Fix

If adaptive memory is making performance worse, you can disable it:

### Option 1: Environment Variable (Recommended)

**Before starting the EDON server:**
```bash
export EDON_DISABLE_ADAPTIVE_MEMORY=1
python -m app.main
```

**Or in Windows PowerShell:**
```powershell
$env:EDON_DISABLE_ADAPTIVE_MEMORY="1"
python -m app.main
```

### Option 2: Clear Memory Database

If you want to start fresh:
```python
from app.robot_stability_memory import get_robot_stability_memory
memory = get_robot_stability_memory()
memory.clear()
```

### Option 3: Delete Database File

```bash
# Delete the memory database
rm data/robot_stability_memory.db
```

## Why Disable It?

Adaptive memory needs **200+ records** before it starts adjusting. Until then:
- It should use base modulations (no adjustments)
- But if there's a bug, it might still adjust incorrectly

**If you're seeing worse performance:**
1. **Disable adaptive memory** (use base v8 policy only)
2. **You should get your previous 50-80% improvement back**
3. **After collecting 200+ records**, you can re-enable it

## Current Settings (Very Conservative)

- **Minimum records**: 200 (was 100)
- **Strategy samples**: 50 per strategy (was 30)
- **Risk baseline**: 100 records (was 50)
- **Small adjustments**: 0.95x-1.05x range

Even with these conservative settings, if it's still making things worse, **disable it** until we can debug further.

