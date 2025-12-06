# V4 Performance Issue - Analysis

## Problem Statement

V4 configuration shows **consistent negative performance** across all validation tests:
- Multi-seed (5 seeds): **-1.7% ± 1.6%**
- 100-episode: **-1.7%**
- Re-run 30-episode: **-1.7%**

## Current Configuration

```python
EDON_V31_HS_V4_CFG = {
    "BASE_GAIN": 0.5,
    "INSTABILITY_WEIGHT": 0.4,
    "DISTURBANCE_WEIGHT": 0.2,
    "RECOVERY_BOOST": 1.2,
    "PREFALL_MIN": 0.15,
    "PREFALL_RANGE": 0.50,
    "PREFALL_MAX": 0.70,
    "SAFE_THRESHOLD": 0.75,
    "SAFE_GAIN": 0.12,
}
```

## Possible Root Causes

### 1. Adaptive Gain Too Aggressive
- **BASE_GAIN=0.5** may be too high
- Combined with instability/disturbance weights, effective gain may exceed 1.0
- **Hypothesis**: Over-correction causing instability

### 2. PREFALL Corrections Destabilizing
- **PREFALL_RANGE=0.50** means corrections up to 0.65 (65% of baseline)
- May be fighting baseline controller instead of helping
- **Hypothesis**: Corrections in wrong direction or too large

### 3. State Mapping Issues
- EDON state may not be correctly mapped
- Instability/disturbance scores may be incorrect
- **Hypothesis**: Wrong state → wrong corrections

### 4. Recovery Boost Too High
- **RECOVERY_BOOST=1.2** adds 20% extra during recovery
- May cause overshoot
- **Hypothesis**: Recovery corrections too aggressive

## Recommended Investigation Steps

1. **Debug state mapping**: Log actual EDON state values
2. **Test lower BASE_GAIN**: Try 0.3, 0.35, 0.4
3. **Test lower PREFALL_RANGE**: Try 0.35, 0.40
4. **Test without recovery boost**: Set RECOVERY_BOOST=1.0
5. **Check correction direction**: Verify corrections oppose tilt

## Next Actions

1. Create V5 configuration with reduced gains
2. Run incremental tests with lower parameters
3. Compare V4 vs V5 to identify what's causing the issue
4. Consider reverting to simpler configuration if needed

