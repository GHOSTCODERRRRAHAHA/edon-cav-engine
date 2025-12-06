# V5.1 LPF Adjustment

## Initial V5.1 Results (with 0.85 - 0.3*instability)

**High stress V5.1**: ~-3.3% ± 2.4% avg improvement
- **Worse than V4**: V4 was -1.7% ± 1.6%
- **Conclusion**: LPF alpha formula too aggressive (too much smoothing)

## Adjustment

**Changed alpha formula**:
- **Before**: `0.85 - 0.3 * instability` (85% → 55% smoothing)
- **After**: `0.8 - 0.2 * instability` (80% → 60% smoothing)

**Rationale**:
- Lower overall smoothing (80% vs 85% when stable)
- Less aggressive instability scaling (0.2 vs 0.3)
- Should reduce over-smoothing that's causing negative results

## Expected Impact

- Less smoothing overall → more responsive
- Still filters noise but not as aggressively
- Should improve from -3.3% toward positive territory

## Testing

Running 5 seeds × 30 episodes with adjusted alpha.

**Target**: +1% to +3% (consistent sign)

If still negative, may need:
- Even lower alpha (0.75 - 0.15 * instability)
- Or different approach to LPF

