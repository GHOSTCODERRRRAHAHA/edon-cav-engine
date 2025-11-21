# Adaptive Memory & Memory Engine Status

## ✅ YES - We Have Both!

### 1. **Adaptive Memory Engine** (`app/adaptive_memory.py`)

**Features:**
- ✅ Rolling 24-hour context buffer
- ✅ SQLite persistence (`data/memory.db`)
- ✅ Hourly EWMA statistics (mean, variance, state distributions)
- ✅ Adaptive adjustments:
  - `z_cav`: Z-score of current CAV relative to baseline
  - `sensitivity`: Sensitivity multiplier (1.0 = normal, >1.0 = increased)
  - `env_weight_adj`: Environment weight adjustment (1.0 = normal, <1.0 = reduced)

**How it works:**
1. Records each CAV response with timestamp, state, environmental data
2. Maintains rolling 24-hour buffer (in-memory + SQLite)
3. Computes hourly statistics using EWMA (Exponential Weighted Moving Average)
4. Calculates adaptive adjustments based on:
   - Current CAV vs. hourly baseline (Z-score)
   - Environmental conditions (AQI, time of day)
   - Historical patterns

### 2. **Memory Management Routes** (`app/routes/memory.py`)

**Endpoints:**
- ✅ `GET /memory/summary` - Get 24-hour memory statistics
- ✅ `POST /memory/clear` - Clear all memory (for testing)

**Returns:**
- Total records in last 24h
- Window size
- Hourly statistics (mean, std, state probabilities)
- Overall statistics

### 3. **Integration Status**

#### ✅ `/cav` Route (Single Window)
- ✅ Records each CAV computation: `memory_engine.record()`
- ✅ Computes adaptive adjustments: `memory_engine.compute_adaptive()`
- ✅ Returns `AdaptiveInfo` in response:
  ```python
  adaptive_info = AdaptiveInfo(
      z_cav=z_score,
      sensitivity=sensitivity_multiplier,
      env_weight_adj=env_weight_adjustment
  )
  ```

#### ⚠️ `/oem/cav/batch` Route (Batch)
- ❌ **NOT currently using adaptive memory**
- The batch route processes windows but doesn't record them or compute adaptive adjustments
- This could be added if needed

### 4. **Response Model**

The `CAVResponse` model includes:
```python
class CAVResponse(BaseModel):
    cav_raw: int
    cav_smooth: int
    state: str
    parts: Dict[str, float]
    adaptive: Optional[AdaptiveInfo]  # ← Adaptive adjustments
```

### 5. **Database**

- ✅ SQLite database: `data/memory.db`
- ✅ Stores: timestamp, cav_raw, cav_smooth, state, parts, temp_c, humidity, aqi, local_hour
- ✅ Automatic cleanup of records older than 24 hours

### 6. **Usage Example**

When you call `POST /cav`, the response includes:
```json
{
  "cav_raw": 5234,
  "cav_smooth": 5123,
  "state": "balanced",
  "parts": {
    "bio": 0.4,
    "env": 0.2,
    "circadian": 0.2,
    "p_stress": 0.2
  },
  "adaptive": {
    "z_cav": 0.15,        // Current CAV is 0.15 std devs above baseline
    "sensitivity": 1.05,   // 5% increased sensitivity
    "env_weight_adj": 0.95 // 5% reduced environment weight
  }
}
```

## Summary

✅ **Adaptive Memory Engine**: Fully implemented and working  
✅ **Memory Routes**: Available at `/memory/summary` and `/memory/clear`  
✅ **Single CAV Route**: Fully integrated with adaptive memory  
⚠️ **Batch Route**: Not using adaptive memory (could be added)

## To Add Adaptive Memory to Batch Route

If you want batch processing to also use adaptive memory, we would need to:
1. Import `AdaptiveMemoryEngine` in `batch.py`
2. Record each window result: `memory_engine.record(...)`
3. Compute adaptive adjustments: `memory_engine.compute_adaptive(...)`
4. Include adaptive info in `BatchResponseItem` (would need to update model)

