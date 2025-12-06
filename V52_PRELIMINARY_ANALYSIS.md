# V5.2 Predicted Boost - Preliminary Analysis

## Current Results (3 seeds each, medium_stress)

| Test | Boost Mult | Scale | Cap | Max Boost | Mean | Status |
|------|-----------|-------|-----|-----------|------|--------|
| 1 | 0.15 | 1.5 | 0.5 | 7.5% | -2.1% | ❌ Too conservative |
| 2 | 0.20 | 2.0 | 0.5 | 10% | -1.4% | ❌ Current V5.2 |
| 3 | 0.25 | 2.5 | 0.5 | 12.5% | -0.0% (1/3) | ⏳ Running |
| 4 | 0.20 | 1.5 | 0.5 | 7.5% | [Running...] | ⏳ |
| 5 | 0.20 | 2.0 | 0.6 | 12% | [Running...] | ⏳ |

## Observations

1. **Test 1** (very conservative): -2.1% - Too little boost
2. **Test 2** (current V5.2): -1.4% - Still negative
3. **Test 3** (slightly more): -0.0% so far - Neutral, may be better

## Hypothesis

- Predicted boost may not be helping on medium_stress
- Or delta_ema may not be a good proxy for predicted instability
- May need different approach or different signal

## Next Steps

1. ⏳ Wait for all tests to complete
2. ⏳ If all negative → consider:
   - Different predicted instability source
   - Remove predicted boost, keep only LPF
   - Try different profiles (light_stress, hell_stress)

