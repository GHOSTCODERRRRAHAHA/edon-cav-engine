# Sweet Spot Search - Final Configuration

## Test Results Summary

| Test | Config | Interventions | Stability | Average | Notes |
|------|--------|---------------|-----------|---------|-------|
| 1 | KP=0.14 | +1.1% | +0.3% | **+0.7%** | Too weak |
| 2 | KP=0.14, PREFALL=0.30 | +2.5% | -5.4% | **-1.5%** | Tradeoff |
| 3 | KP=0.16, PREFALL=0.32 | +1.6% | +0.4% | **+1.0%** | Best so far |
| 4 | KP=0.16, PREFALL=0.36, SAFE=0.025 | -1.3% | -2.5% | **-1.9%** | SAFE too strong |
| 5 | KP=0.18, PREFALL=0.34, SAFE minimal | [Testing...] | [Testing...] | [TBD] | Current test |

## Current Configuration (Test 5)
- **KP_ROLL/PITCH**: 0.18 (80% above baseline) - Stronger PD
- **KD_ROLL/PITCH**: 0.05 (67% above baseline)
- **PREFALL_BASE**: 0.34 (34% base) - Between Test 3 (0.32) and Test 4 (0.36)
- **PREFALL_MIN**: 0.20
- **PREFALL_MAX**: 0.46
- **SAFE_GAIN**: 0.015 (minimal, dialed back)
- **SAFE_CORR**: -0.06 (minimal, dialed back)

## Strategy
- Test 3 gave +1.0% - good direction
- Test 4 with stronger SAFE made it worse - SAFE corrections problematic
- Test 5: Stronger PD (0.18) + moderate PREFALL (0.34) + minimal SAFE
- If still <10%, may need to:
  1. Increase PREFALL to 0.36-0.38
  2. Try different EDON gains (0.60, 0.90, 1.00)
  3. Adjust mode multipliers
  4. Check direction of corrections

## Target
**10%+ average improvement** in both interventions AND stability

