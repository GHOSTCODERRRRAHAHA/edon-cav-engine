# Incremental Testing - Finding Sweet Spot for 10%+

## Strategy
Instead of aggressive 2-3x increases, test **moderate incremental increases** to find the sweet spot.

## Current Configuration (After Dialing Back)
- **PD Gains**: KP=0.14 (40% increase), KD=0.04 (33% increase)
- **PREFALL_BASE**: 0.30 (30% base, was 0.28)
- **SAFE_GAIN**: 0.020 (2%, was 0.015)
- **SAFE correction**: -0.08 (60% increase, was -0.05)
- **Mode multipliers**: Moderate increases (30-40% boosts)

## Test Plan

### Test 1: Moderate PD Only ✓
- KP=0.14, KD=0.04
- PREFALL_BASE=0.28 (unchanged)
- SAFE_GAIN=0.015 (unchanged)
- **Result**: [Running...]

### Test 2: Moderate PD + PREFALL
- KP=0.14, KD=0.04
- PREFALL_BASE=0.30
- SAFE_GAIN=0.015 (unchanged)
- **Result**: [Pending]

### Test 3: Moderate PD + PREFALL + SAFE
- KP=0.14, KD=0.04
- PREFALL_BASE=0.30
- SAFE_GAIN=0.020, SAFE_CORR=-0.08
- **Result**: [Pending]

### Test 4: Stronger PD + PREFALL + SAFE
- KP=0.16, KD=0.045
- PREFALL_BASE=0.30
- SAFE_GAIN=0.020, SAFE_CORR=-0.08
- **Result**: [Pending]

### Test 5: Stronger PD + Stronger PREFALL + SAFE
- KP=0.16, KD=0.045
- PREFALL_BASE=0.32
- SAFE_GAIN=0.020, SAFE_CORR=-0.08
- **Result**: [Pending]

## Incremental Approach
1. Start with moderate increases (40-50%)
2. Test each combination
3. If <10%, increase next parameter
4. If oscillations occur, dial back
5. Find sweet spot where improvements are 10%+ without destabilizing

## Files
- `incremental_test.py` - Automated testing script
- `evaluation/edon_controller_v3.py` - Current moderate settings

