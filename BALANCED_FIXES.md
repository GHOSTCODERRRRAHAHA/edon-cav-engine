# Balanced Fixes - Targeting 10% Without Destabilizing

## Problem
Aggressive fixes made things WORSE (-4.2% average). Corrections are too strong and causing oscillations.

## Strategy
Find the **sweet spot**: Strong enough to achieve 10%+, but not so strong it destabilizes.

## Current Values (After Aggressive Fixes)
- PD Gains: KP=0.18, KD=0.05 (80% increase)
- PREFALL_BASE: 0.35 (35% base)
- SAFE_GAIN: 0.025 (2.5%)
- SAFE correction: -0.12 (2.4x)

## Balanced Approach
Instead of 2-3x increases everywhere, use **moderate increases** with **better tuning**:

1. **Moderate PD gains**: 0.10 → 0.14 (40% increase, not 80%)
2. **Moderate PREFALL**: 0.28 → 0.32 (32% base, not 35%)
3. **Moderate SAFE**: 0.015 → 0.020 (2%, not 2.5%)
4. **Better direction checking**: Ensure corrections always stabilize
5. **Add damping to corrections**: Prevent oscillations

## Next Steps
1. Dial back aggressive values to moderate levels
2. Add correction damping
3. Improve direction checking
4. Test incrementally

