# V4 Multi-Seed Validation Results

## Configuration Tested
- **BASE_GAIN**: 0.5
- **PREFALL_RANGE**: 0.50
- **PREFALL_MAX**: 0.70
- **Profile**: high_stress
- **N_SEEDS**: 5
- **Episodes per seed**: 30

## Results

### Individual Seed Results

| Seed | Interventions | Stability | Average |
|------|---------------|-----------|---------|
| 1 | -2.6% | -1.9% | **-2.2%** |
| 2 | +0.0% | +0.6% | **+0.3%** |
| 3 | +3.6% | -4.2% | **-0.3%** |
| 4 | -0.4% | -3.5% | **-1.9%** |
| 5 | [Not shown] | [Not shown] | **-4.3%** |

### Summary Statistics

**High stress V4: ~-1.7% ± 1.6% avg improvement**
- **Mean**: -1.7%
- **Std**: 1.6%
- **Range**: -4.3% to +0.3%
- **Based on**: 5 seeds, 30 episodes each

## Analysis

### Key Findings

1. **Consistent negative performance**: 4 out of 5 seeds show negative improvement
2. **High variance**: Range from -4.3% to +0.3% (±2.3% swing)
3. **Confirms earlier results**: Matches 100-episode test (-1.7%)
4. **Configuration issue**: V4 is not providing stable improvement

### Comparison with Earlier Tests

| Test | Episodes | Mean | Status |
|------|----------|------|--------|
| Original 30-ep | 30 | +6.4% | ❌ Variance (lucky roll) |
| Re-run 30-ep | 30 | -1.7% | ❌ Confirmed negative |
| 100-ep | 100 | -1.7% | ❌ Consistent negative |
| Multi-seed (5 seeds) | 150 total | **-1.7% ± 1.6%** | ❌ **Confirmed negative** |

## Conclusion

**V4 Configuration: NOT VALIDATED for performance claims**

- **Stable band**: -1.7% ± 1.6% (consistently worse than baseline)
- **Variance**: High (±1.6% std, range of ±2.3%)
- **Status**: Configuration is locked but performing poorly

## Next Steps

1. **Investigate root cause**: Why is V4 performing worse?
   - Adaptive gain formula may be too aggressive
   - PREFALL corrections may be destabilizing
   - State mapping may be incorrect

2. **Try different parameters**:
   - Lower BASE_GAIN (0.3-0.4)
   - Reduce PREFALL_RANGE (0.35-0.40)
   - Adjust state thresholds

3. **Consider reverting**: May need to go back to simpler configuration

4. **Test other profiles**: Check if issue is specific to high_stress

## Configuration Status

- ✅ **Locked**: Configuration is frozen as reference
- ❌ **Performance**: Consistently worse than baseline
- ⚠️ **Use**: Reference baseline for future improvements (even if negative)

