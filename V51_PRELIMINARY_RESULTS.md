# V5.1 Incremental LPF - Preliminary Results

## Current Status

**Test 1** (0.75 - 0.15*instability): **+0.2% ± 0.2%** ✅
- All 3 seeds complete
- **Slightly positive** - LPF is helping!

**Test 2** (0.80 - 0.20*instability): -0.5% (1/3 seeds)
- Still running

**Tests 3-5**: Still running

## Early Analysis

Test 1 shows **+0.2%** which is:
- ✅ **Positive** (better than V4's -1.7%)
- ✅ **Low variance** (±0.2%)
- ⚠️ **Below target** (+1% to +3%)

## Next Steps

1. ⏳ Wait for all tests to complete
2. ⏳ Compare all 5 formulas
3. ⏳ If Test 1 is best → use it, run full 5-seed validation
4. ⏳ If another test is better → use that one
5. ⏳ If all are slightly positive → proceed to V5.2 (add predicted boost)

## Observation

**Test 1 (0.75 - 0.15) is working!**
- Lower base (75% vs 80-85%) seems better
- Less aggressive scaling (0.15 vs 0.20-0.30) seems better
- This suggests we need less smoothing overall

