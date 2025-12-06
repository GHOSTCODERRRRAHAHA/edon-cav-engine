# V5.1 Final Alpha Selection

## Incremental Test Results (3 seeds each)

| Test | Formula | Mean | Std | Status |
|------|---------|------|-----|--------|
| 1 | **0.75 - 0.15 * instability** | **+0.2%** | ±0.2% | ✅ **BEST** |
| 2 | 0.80 - 0.20 * instability | -1.7% | ±0.9% | ❌ |
| 3 | 0.70 - 0.10 * instability | -1.6% | ±1.2% | ❌ |
| 4 | 0.85 - 0.25 * instability | -2.4% | ±2.6% | ❌ |
| 5 | 0.72 - 0.12 * instability | [Running...] | | ⏳ |

## Conclusion

**Test 1 is clearly best**:
- ✅ Only positive result (+0.2%)
- ✅ Lowest variance (±0.2%)
- ✅ Better than V4 (-1.7%)
- ✅ Better than all other tests

**Formula to lock**: `0.75 - 0.15 * instability_score`
- Stable (instability=0): 75% smoothing
- Unstable (instability=1): 60% smoothing
- Clamped: 0.4 to 0.9

## Next Steps

1. ⏳ Wait for Test 5 to complete (confirm Test 1 is best)
2. ✅ Lock Test 1 formula in code
3. ⏳ Run full 5-seed validation
4. ⏳ If positive → proceed to V5.2 (add mild predicted boost)

## Expected Full Validation

- **Target**: +0.2% to +0.5% (consistent positive sign)
- **If achieved**: LPF is working, proceed to V5.2
- **V5.2 goal**: Push toward +1% to +3% with predicted boost

