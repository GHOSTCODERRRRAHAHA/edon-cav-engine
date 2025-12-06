# V5.1 Best Alpha Formula Selection

## Current Results (3 seeds each)

| Test | Formula | Mean | Std | Status |
|------|---------|------|-----|--------|
| 1 | 0.75 - 0.15 * instability | **+0.2%** | ±0.2% | ✅ **BEST** |
| 2 | 0.80 - 0.20 * instability | -1.7% | ±0.9% | ❌ Worse |
| 3 | 0.70 - 0.10 * instability | -0.5% | [Running...] | ⏳ |
| 4 | 0.85 - 0.25 * instability | [Running...] | | ⏳ |
| 5 | 0.72 - 0.12 * instability | [Running...] | | ⏳ |

## Analysis

**Test 1 is currently best**:
- ✅ Positive (+0.2%)
- ✅ Low variance (±0.2%)
- ✅ Better than V4 (-1.7%)
- ⚠️ Below target (+1% to +3%)

**Test 2 is worse**:
- Higher base (0.80 vs 0.75) → too much smoothing
- Higher scale (0.20 vs 0.15) → too aggressive

## Next Steps

1. ⏳ Wait for all tests to complete
2. ⏳ Confirm Test 1 is best
3. ⏳ Lock Test 1 formula: `0.75 - 0.15 * instability`
4. ⏳ Run full 5-seed validation
5. ⏳ If still positive → proceed to V5.2 (add predicted boost)

## Expected Outcome

- **Test 1** should remain best (lower base = less smoothing = better)
- **Full validation** should confirm +0.2% to +0.5% range
- **V5.2** (add predicted boost) should push toward +1% to +3% target

