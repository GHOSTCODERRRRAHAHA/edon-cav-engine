# V4 Incremental Testing Summary

## Goal
Find configuration that consistently achieves +6.4% or better improvement.

## Test Results

### Best Single Run
- **Test 1** (BASE_GAIN=0.5): **+7.0%** average
  - Interventions: -0.2%
  - Stability: +14.1%
  - **Status**: Exceeds target, but high variance

### Most Consistent
- **Test 8** (BASE_GAIN=0.5, PREFALL_RANGE=0.50): **+4.7%** average
  - Interventions: -0.7%
  - Stability: +10.2%
  - **Status**: Below target but more consistent

### Current Final Config
- **BASE_GAIN=0.5, PREFALL_RANGE=0.50**: **+2.9%** average (latest run)
  - Interventions: -1.1%
  - Stability: +6.9%
  - **Status**: Below target, high variance

## Key Findings

1. **BASE_GAIN=0.5** is better than 0.4
2. **PREFALL_RANGE=0.50** provides better stability improvements
3. **High variance**: 30-episode tests show ±3-4% swings
4. **Best result**: +7.0% (but not reproducible)
5. **Consistent range**: +2.9% to +4.7% (below +6.4% target)

## Variance Analysis

| Test | Run 1 | Run 2 | Run 3 | Average | Range |
|------|-------|-------|-------|---------|-------|
| Test 1 (BASE_GAIN=0.5) | +7.0% | -1.8% | [Not tested] | ~+2.6% | ±4.4% |
| Test 8 (BASE_GAIN=0.5, PREFALL=0.50) | +4.7% | +2.9% | [Not tested] | ~+3.8% | ±0.9% |

## Recommendations

### Option 1: Accept High Variance
- Use **BASE_GAIN=0.5, PREFALL_RANGE=0.50**
- Run multiple 30-episode tests and average
- Target: Average of 3-5 runs should be +4-6%

### Option 2: Focus on 100-Episode Tests
- 30-episode tests have too much variance
- Run 100-episode tests with best config
- More reliable but slower

### Option 3: Further Tuning
- Try BASE_GAIN=0.55-0.60
- Adjust PREFALL_RANGE to 0.55
- Test recovery boost increases

## Current V4 Configuration (Updated)

```python
EDON_V31_HS_V4_CFG = {
    "BASE_GAIN": 0.5,  # Updated from 0.4
    "INSTABILITY_WEIGHT": 0.4,
    "DISTURBANCE_WEIGHT": 0.2,
    "RECOVERY_BOOST": 1.2,
    "PREFALL_MIN": 0.15,
    "PREFALL_RANGE": 0.50,  # Updated from 0.45
    "PREFALL_MAX": 0.70,  # Updated from 0.65
    "SAFE_THRESHOLD": 0.75,
    "SAFE_GAIN": 0.12,
}
```

## Status

- ✅ **Configuration updated**: BASE_GAIN=0.5, PREFALL_RANGE=0.50
- ⚠️ **Target not met**: Best consistent result is +4.7% (below +6.4%)
- ⚠️ **High variance**: 30-episode tests show large swings
- ⏳ **Next step**: Run 100-episode validation or multiple 30-episode runs

