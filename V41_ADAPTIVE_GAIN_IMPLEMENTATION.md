# V4.1 Adaptive Gain System Implementation

## Summary

Replaced the old gain logic in `apply_edon_regulation()` with the new V4.1 adaptive gain system based on the top-performing configurations (+7.0% and +6.4% improvements).

## Changes Made

### 1. Adaptive EDON Gain (V4.1)

**Replaced**: Old target_ratio-based system with complex state modulation

**New System**:
```python
BASE_GAIN = 0.50  # Was 0.40 → better stability backbone

# Instability score (weighted stress/chaos)
instability_score = 0.6 * p_chaos + 0.4 * p_stress

# Disturbance proxy (risk EMA)
disturbance = risk_ema

# Raw adaptive gain
gain = BASE_GAIN + 0.40 * instability_score + 0.20 * disturbance

# Recovery boost
if tilt_zone == "FAIL":
    gain *= 1.20  # 20% stronger during recovery
elif prefall_zone == "ATTN":
    gain *= 1.10  # mild boost

# Clamp gain range
gain = max(0.30, min(gain, 1.10))
```

**Key Features**:
- BASE_GAIN increased from 0.40 to 0.50 (based on +7.0% result)
- Uses p_chaos and p_stress directly from EDON output
- Recovery boost: 20% in FAIL zone, 10% in prefall
- Gain range: 0.30 to 1.10

### 2. Dynamic PREFALL Reflex

**Replaced**: Old constant PREFALL scaling

**New System**:
```python
PREFALL_MIN = 0.15
PREFALL_MAX = 0.60

# fall_risk ranges 0-1 (using risk_ema)
prefall_gain = PREFALL_MIN + (PREFALL_MAX - PREFALL_MIN) * fall_risk
prefall_gain = min(PREFALL_MAX, max(PREFALL_MIN, prefall_gain))

# Only apply in prefall/fail zones
if internal_zone in ("prefall", "fail"):
    prefall_torque *= prefall_gain
else:
    prefall_torque *= 0.0  # disabled in safe zone
```

**Key Features**:
- Dynamic scaling: 0.15 (low risk) to 0.60 (high risk)
- Only active in prefall/fail zones
- Uses risk_ema as fall_risk proxy

### 3. Catastrophic-Only SAFE Reflex

**Replaced**: Old continuous SAFE corrections

**New System**:
```python
if catastrophic_risk > 0.75:  # 75% danger threshold
    SAFE_GAIN = 0.12  # previously too high/too frequent
    safe_torque[0] = -0.15 * roll * SAFE_GAIN
    safe_torque[1] = -0.15 * pitch * SAFE_GAIN
else:
    safe_torque *= 0.0  # disabled unless catastrophic
```

**Key Features**:
- Only activates when catastrophic_risk > 0.75
- SAFE_GAIN = 0.12 (reduced from previous values)
- Applied only to balance joints (indices 0-3)

### 4. Final Action Blend (V4.1)

**Replaced**: Old blending with edon_gain parameter

**New System**:
```python
final_action = (
    baseline_action
    + prefall_torque * adaptive_gain  # Use adaptive gain
    + safe_torque
)
```

**Key Features**:
- Uses adaptive_gain instead of fixed edon_gain
- prefall_torque already scaled by prefall_gain
- Total scaling: correction * prefall_gain * adaptive_gain

### 5. Debug Fields

**Added**:
```python
debug_info["adaptive_gain"] = gain
debug_info["prefall_gain"] = prefall_gain if internal_zone in ("prefall", "fail") else 0.0
debug_info["safe_active"] = safe_active
```

## Implementation Details

### Variable Extraction
- `p_chaos`: From `edon_state_raw.get("p_chaos", 0.0)`
- `p_stress`: From `edon_state_raw.get("p_stress", 0.0)`
- `fall_risk`: Uses `risk_ema` as proxy
- `catastrophic_risk`: `1.0` if in FAIL zone, else `risk_ema * 1.2` (clamped to [0, 1])

### Zone Detection
- Uses `internal_zone` (with virtual prefall) for gain computation
- Uses `tilt_zone` for metrics (unchanged)

### State Storage
- `controller_state["adaptive_gain"]`: Stores computed gain
- `controller_state["safe_active"]`: Stores whether SAFE is active

## Expected Improvements

Based on historical results:
- **V4 (BASE_GAIN=0.4)**: +6.4% average
- **Test 1 (BASE_GAIN=0.5)**: +7.0% average (best single run)

**V4.1 combines both**:
- BASE_GAIN=0.50 (from +7.0% result)
- Adaptive gain system (from +6.4% result)
- Dynamic PREFALL (from +6.4% result)
- Catastrophic-only SAFE (from +6.4% result)

## Files Modified

- `run_eval.py`: `apply_edon_regulation()` function (lines ~448-572)

## Testing Recommendations

1. Run baseline vs EDON comparison on high_stress profile
2. Check debug output for adaptive_gain, prefall_gain, safe_active
3. Verify gain ranges are within expected bounds (0.30-1.10)
4. Confirm PREFALL only activates in prefall/fail zones
5. Confirm SAFE only activates when catastrophic_risk > 0.75

## Next Steps

1. Validate performance matches or exceeds V4 (+6.4%)
2. Test across all profiles (light, medium, high, hell)
3. Fine-tune BASE_GAIN if needed (0.50 may need adjustment)
4. Optimize PREFALL_MIN/MAX if needed (currently 0.15-0.60)





