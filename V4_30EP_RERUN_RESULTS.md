# EDON v3.1 High-Stress V4 - 30 Episode Re-Run Results

## Purpose
Re-run 30-episode test to check if original +6.4% result was variance or consistent.

## Results

### Baseline (30 episodes, re-run)
- **Interventions**: 40.0/ep
- **Stability**: 0.0213

### EDON v3.1 HS V4 (30 episodes, re-run)
- **Interventions**: 42.4/ep
- **Stability**: 0.0208

### Improvements (Re-run)
- **Interventions**: -5.8% (worse)
- **Stability**: +2.5% (better)
- **Average**: **-1.7%** (worse)

## Comparison Table

| Test | Episodes | Interventions | Stability | Average | Status |
|------|----------|---------------|-----------|---------|--------|
| Original 30-ep | 30 | +4.4% | +8.4% | **+6.4%** | ❌ Variance |
| Re-run 30-ep | 30 | -5.8% | +2.5% | **-1.7%** | ❌ Confirmed |
| 100-ep | 100 | -0.3% | -3.0% | **-1.7%** | ❌ Consistent |

## Analysis

### Conclusion: Original +6.4% was variance
- **Original 30-ep**: +6.4% (lucky roll)
- **Re-run 30-ep**: -1.7% (matches 100-ep)
- **100-ep**: -1.7% (consistent)

### Key Findings
1. **High variance**: 30-episode tests show large variance (±6-8%)
2. **Consistent negative**: Both re-run and 100-ep show -1.7% average
3. **Configuration issue**: V4 is not providing stable improvement
4. **Baseline variance**: Baseline also varies (40.0-41.4 int/ep)

### Status
- ❌ **V4 NOT validated**: Configuration shows negative improvement
- ❌ **Original result was variance**: +6.4% was not reproducible
- ✅ **Configuration locked**: V4 is frozen as reference (even if negative)

## Next Steps
1. Investigate why V4 performs worse than baseline
2. Consider reverting to previous configuration
3. Test different adaptive gain parameters
4. Check if issue is specific to high_stress profile

