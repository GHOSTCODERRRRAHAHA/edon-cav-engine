# V5.1 Configuration - LOCKED

## Alpha Formula Selected

**Formula**: `0.75 - 0.15 * instability_score`
- **Stable** (instability=0): 75% smoothing
- **Unstable** (instability=1): 60% smoothing
- **Clamped**: 0.4 to 0.9

## Selection Process

Tested 5 alpha formulas with 3 seeds each:

| Test | Formula | Mean | Status |
|------|---------|------|--------|
| 1 | **0.75 - 0.15** | **+0.2%** | ✅ **SELECTED** |
| 2 | 0.80 - 0.20 | -1.7% | ❌ |
| 3 | 0.70 - 0.10 | -1.6% | ❌ |
| 4 | 0.85 - 0.25 | -2.4% | ❌ |
| 5 | 0.72 - 0.12 | -1.8% | ❌ |

## Why Test 1 Won

- ✅ **Only positive result** (+0.2%)
- ✅ **Lowest variance** (±0.2%)
- ✅ **Better than V4** (-1.7%)
- ✅ **Better than all alternatives**

## Full Validation

Running 5 seeds × 30 episodes to confirm:
- **Target**: +0.2% to +0.5% (consistent positive sign)
- **If achieved**: LPF is working, proceed to V5.2

## Next Steps

1. ⏳ Complete full 5-seed validation
2. ⏳ If positive → proceed to V5.2 (add mild predicted boost)
3. ⏳ V5.2 goal: Push toward +1% to +3%

## Status

- ✅ **Alpha formula locked**: 0.75 - 0.15 * instability
- ⏳ **Full validation**: Running (5 seeds)
- ⏳ **V5.2**: Pending (add predicted boost)

