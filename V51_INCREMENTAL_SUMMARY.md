# V5.1 Incremental LPF Fix - Summary

## Problem
- **V4**: -1.7% ± 1.6% (baseline, no LPF)
- **V5.1 initial**: -3.3% ± 2.4% (alpha = 0.85 - 0.3*instability, too much smoothing)

## Solution: Incremental Alpha Testing

### Current Status
- **Test 1** (0.75 - 0.15*instability): +0.0% with 2/3 seeds complete
- **Tests 2-5**: Still running

### Test Matrix (3 seeds × 30 episodes each)

| Test | Formula | Stable Alpha | Unstable Alpha | Status |
|------|---------|--------------|----------------|--------|
| 1 | 0.75 - 0.15 * instability | 75% | 60% | ⏳ 2/3 seeds |
| 2 | 0.80 - 0.20 * instability | 80% | 60% | ⏳ Running |
| 3 | 0.70 - 0.10 * instability | 70% | 60% | ⏳ Running |
| 4 | 0.85 - 0.25 * instability | 85% | 60% | ⏳ Running |
| 5 | 0.72 - 0.12 * instability | 72% | 60% | ⏳ Running |

### Process

1. **Incremental testing**: Test 5 alpha formulas with 3 seeds each (faster iteration)
2. **Select best**: Choose formula with most positive mean
3. **Full validation**: Run 5 seeds with best formula
4. **Proceed**: If positive → V5.2 (add mild predicted boost)

### Target

**Goal**: +1% to +3% with consistent sign

### Next Steps

1. ⏳ Complete incremental tests (all 5 formulas, 3 seeds each)
2. ⏳ Select best alpha formula
3. ⏳ Run full 5-seed validation with best formula
4. ⏳ If positive → proceed to V5.2
5. ⏳ If still negative → try even lower alpha (0.65-0.70 base)

## Files

- **Test script**: `test_lpf_incremental.py`
- **Progress checker**: `check_lpf_progress.py`
- **Current alpha**: `0.75 - 0.15 * instability` (test 1, in code)

