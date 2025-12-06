# EDON v3.1 High-Stress V4 - Final Configuration

## Updated Parameters

Based on incremental testing, V4 configuration has been updated:

1. **BASE_GAIN**: 0.4 → **0.5** (increased)
2. **PREFALL_RANGE**: 0.45 → **0.50** (increased)
3. **PREFALL_MAX**: 0.65 → **0.70** (increased to match range)

## Incremental Test Results

| Test | Config | Average | Status |
|------|--------|---------|--------|
| Test 1 | BASE_GAIN=0.5 | +7.0% | ✅ Best (but high variance) |
| Test 8 | BASE_GAIN=0.5, PREFALL_RANGE=0.50 | +4.7% | ⚠️ Most consistent |
| Test 6 | BASE_GAIN=0.4 (original) | +3.1% | ❌ |

## Final V4 Configuration

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

## Key Findings

1. **BASE_GAIN=0.5** is better than 0.4
2. **PREFALL_RANGE=0.50** provides more consistent results
3. **High variance** in 30-episode tests (±3-4% swings)
4. **Best single run**: +7.0% (BASE_GAIN=0.5)
5. **Most consistent**: +4.7% (BASE_GAIN=0.5, PREFALL_RANGE=0.50)

## Validation Status

- ⏳ **30-episode validation**: [Testing...]
- ⏳ **100-episode validation**: [Pending]

## Note on Variance

30-episode tests show high variance. The original +6.4% and test1's +7.0% may be variance. The updated configuration (BASE_GAIN=0.5, PREFALL_RANGE=0.50) shows more consistent +4.7% results, which is closer to the target but still below +6.4%.

