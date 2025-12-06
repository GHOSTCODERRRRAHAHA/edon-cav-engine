# V5.2 Current Status

## Incremental Testing Progress

**Profile**: medium_stress  
**Tests**: 5 configurations, 3 seeds each

## Results So Far

| Test | Config | Mean | Seeds | Status |
|------|--------|------|-------|--------|
| 1 | Boost 0.15, Scale 1.5 | -2.1% | 3/3 | ❌ Complete |
| 2 | Boost 0.20, Scale 2.0 (V5.2) | -1.4% | 3/3 | ❌ Complete |
| 3 | Boost 0.25, Scale 2.5 | [Running...] | | ⏳ |
| 4 | Boost 0.20, Scale 1.5 | [Running...] | | ⏳ |
| 5 | Boost 0.20, Scale 2.0, Cap 0.6 | [Running...] | | ⏳ |

## Findings

- **Test 1** (very conservative): -2.1% - Too little boost
- **Test 2** (current V5.2): -1.4% - Still negative
- **Test 3**: Still running, showing neutral so far

## Next Steps

1. ⏳ Wait for all tests to complete
2. ⏳ Analyze which configuration is best
3. ⏳ If all negative → consider:
   - Remove predicted boost, keep only LPF (V5.1 was +0.2%)
   - Try different predicted instability source
   - Test on different profiles

## V5.1 Reference

**V5.1** (LPF only, no predicted boost): +0.2% ± 0.2% (3 seeds, high_stress)
- This was positive!
- May be better to keep LPF only and skip predicted boost

