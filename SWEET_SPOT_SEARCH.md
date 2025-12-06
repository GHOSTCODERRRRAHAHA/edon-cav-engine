# Sweet Spot Search - Incremental Testing Results

## Baseline
- Interventions: 40.7/ep
- Stability: 0.0209

## Test Results Summary

| Test | Configuration | Interventions | Stability | Average | Status |
|------|--------------|---------------|-----------|---------|--------|
| 1 | KP=0.14, KD=0.04 | +1.1% | +0.3% | **+0.7%** | ❌ Need more |
| 2 | KP=0.14, PREFALL=0.30 | +2.5% | -5.4% | **-1.5%** | ❌ Tradeoff |
| 3 | KP=0.16, PREFALL=0.32 | +1.6% | +0.4% | **+1.0%** | ❌ Need more |
| 4 | KP=0.16, PREFALL=0.36, SAFE=0.025/-0.10 | [Testing...] | [Testing...] | [TBD] | ⏳ |

## Current Settings (Test 4)
- **KP_ROLL/PITCH**: 0.16 (60% above baseline)
- **KD_ROLL/PITCH**: 0.045 (50% above baseline)
- **PREFALL_BASE**: 0.36 (36% base correction)
- **PREFALL_MIN**: 0.22 (22% minimum)
- **PREFALL_MAX**: 0.48 (48% maximum)
- **SAFE_GAIN**: 0.025 (2.5%)
- **SAFE_CORR**: -0.10 (2x baseline)

## Observations
- Test 1: Small improvement (+0.7%) - need stronger corrections
- Test 2: Tradeoff (better interventions, worse stability) - PREFALL may be too strong or wrong direction
- Test 3: Better balance (+1.0%) - stronger PD helps
- Test 4: Adding SAFE corrections - should help further

## Next Steps
1. ⏳ Complete Test 4
2. If <10%, increase PREFALL to 0.38-0.40
3. If oscillations, add damping
4. Continue until 10%+ achieved

## Target
**10%+ average improvement** in both interventions AND stability (no tradeoffs)

