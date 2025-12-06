# V5.1 Configuration - Low-Pass Filter Only

## Changes from V4

**ONLY** the stability-weighted low-pass filter is added. All other V4 behavior is unchanged:
- ✅ Low-pass filter added
- ❌ No predicted instability pre-boost
- ❌ No PREFALL decay
- ❌ No SAFE pre-trigger

## Low-Pass Filter Implementation

```python
alpha = clamp(0.85 - 0.3 * instability_score, 0.4, 0.9)
torque_cmd = alpha * torque_cmd_prev + (1.0 - alpha) * torque_cmd
```

**Formula**: `0.85 - 0.3 * instability`
- Stable (instability=0): alpha=0.85 (85% smoothing)
- Unstable (instability=1): alpha=0.55 (55% smoothing)
- Clamped: 0.4 to 0.9

## Testing Plan

1. Run 5 seeds × 30 episodes
2. Compute average improvement
3. **If negative**: Adjust LPF alpha formula (e.g., `0.8 - 0.2 * instability`)
4. **If slightly positive (~+1-3%)**: Good, LPF is helping
5. Once LPF-only is not hurting, proceed to V5.2 (add mild predicted boost)

## Expected Results

- **V4**: -1.7% ± 1.6%
- **V5.1 Target**: +1% to +3% (consistent sign)
- **If negative**: Need to adjust alpha formula

## Next Steps

After V5.1 validation:
- If positive: → V5.2 (add mild predicted boost)
- If negative: → Adjust LPF alpha, re-test
- Once consistently positive: → Experiment with PREFALL scaling

