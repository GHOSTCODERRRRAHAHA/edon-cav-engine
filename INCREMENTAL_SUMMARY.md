# Incremental Testing Summary - Sweet Spot Search

## Progress So Far

| Test | Configuration | Average Improvement | Status |
|------|--------------|---------------------|--------|
| 1 | KP=0.14 | +0.7% | Too weak |
| 2 | KP=0.14, PREFALL=0.30 | -1.5% | Tradeoff |
| 3 | KP=0.16, PREFALL=0.32 | +1.0% | Better |
| 4 | KP=0.16, PREFALL=0.36, SAFE=0.025 | -1.9% | SAFE too strong |
| 5 | KP=0.18, PREFALL=0.34 | +1.1% | **Best so far** |
| 6 | KP=0.18, PREFALL=0.40 | [Testing...] | Current |

## Current Best Configuration (Test 5)
- **KP_ROLL/PITCH**: 0.18 (80% above baseline)
- **KD_ROLL/PITCH**: 0.05 (67% above baseline)
- **PREFALL_BASE**: 0.34 (34% base correction)
- **SAFE_GAIN**: 0.015 (minimal)
- **Result**: +1.1% average improvement

## Test 6 Configuration
- **KP_ROLL/PITCH**: 0.18 (same)
- **PREFALL_BASE**: 0.40 (increased from 0.34)
- **PREFALL_MIN**: 0.25 (increased)
- **PREFALL_MAX**: 0.55 (increased)
- **Goal**: Push from +1.1% to 10%+

## Observations
1. **PD gains**: 0.18 seems good (80% increase)
2. **PREFALL**: 0.34 gave +1.1%, trying 0.40 for more
3. **SAFE corrections**: Caused issues when too strong - keeping minimal
4. **Direction**: Need to ensure corrections always stabilize

## Next Steps
- ⏳ Complete Test 6
- If <10%, try:
  - PREFALL=0.42-0.45
  - Different EDON gains (0.60, 0.90, 1.00)
  - Adjust mode multipliers
  - Check correction direction logic

## Target
**10%+ average improvement** in both interventions AND stability

