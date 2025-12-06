# V5.1 Incremental LPF Fix Plan

## Current Status

- **V4**: -1.7% ± 1.6% (baseline, no LPF)
- **V5.1 initial**: -3.3% ± 2.4% (alpha = 0.85 - 0.3*instability, too much smoothing)
- **V5.1a**: Testing (alpha = 0.8 - 0.2*instability)

## Incremental Testing Approach

### Step 1: Test Multiple Alpha Formulas
Running `test_lpf_incremental.py` with 5 different alpha formulas:
1. `0.75 - 0.15 * instability` (75% → 60% smoothing)
2. `0.80 - 0.20 * instability` (80% → 60% smoothing) ← Current V5.1a
3. `0.70 - 0.10 * instability` (70% → 60% smoothing)
4. `0.85 - 0.25 * instability` (85% → 60% smoothing)
5. `0.72 - 0.12 * instability` (72% → 60% smoothing)

**Test**: 3 seeds × 30 episodes each (faster iteration)

### Step 2: Select Best Formula
- Find formula with most positive mean
- If any formula gives +1% to +3% → use it
- If all negative → try even lower alpha (0.65-0.70 base)

### Step 3: Full Validation
- Lock best alpha formula
- Run full 5-seed validation
- Confirm consistent positive sign

### Step 4: Proceed to V5.2
- Once V5.1 is positive → add mild predicted boost
- Test again with 5 seeds
- Only then experiment with PREFALL scaling

## Expected Alpha Behavior

**Lower alpha = less smoothing = more responsive**
- Too low (< 0.6): May not filter noise enough
- Too high (> 0.85): Over-smoothing, delayed response
- **Target range**: 0.70-0.80 when stable, 0.55-0.65 when unstable

## Current Test Status

⏳ Running incremental alpha tests (3 seeds each)...
Results will show which formula works best.

