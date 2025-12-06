# V5.2 Medium Stress Validation

## Configuration

**V5.2 = V5.1 (LPF) + Super-Conservative Predicted Boost**

- **LPF**: `alpha = 0.75 - 0.15 * instability` (unchanged)
- **Predicted instability**: `delta_ema * 2.0`, capped at 0.5
- **Boost**: `gain *= (1.0 + 0.2 * predicted_instability)` (up to +10%)
- **Profile**: **medium_stress** (changed from high_stress)

## Why Medium Stress?

- **Less chaotic**: May show clearer signal
- **More stable baseline**: Easier to see improvements
- **Better for validation**: Less variance, more consistent results

## Testing

Running 5 seeds × 30 episodes on medium_stress profile.

## Expected Results

### Good (Stage 0/early Stage 1)
- Mean: +1% to +3%
- Std: ±1-2%
- Most seeds positive

### Comparison

- **High stress V4**: -1.7% ± 1.6%
- **High stress V5.1**: +0.2% ± 0.2% (3 seeds)
- **Medium stress V5.2**: [Testing...]

## Next Steps

1. ⏳ Complete medium_stress validation (5 seeds)
2. ⏳ Compare with high_stress results
3. ⏳ If positive → proceed to PREFALL scaling
4. ⏳ If still issues → adjust boost multiplier

