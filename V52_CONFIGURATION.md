# V5.2 Configuration - Super-Conservative Predicted Boost

## Summary

**V5.2 = V5.1 (LPF) + Super-Conservative Predicted Instability Boost**

## Components

### 1. Low-Pass Filter (V5.1, unchanged)
- **Formula**: `alpha = 0.75 - 0.15 * instability_score`
- **Result**: +0.2% ± 0.2% (3 seeds)
- **Status**: Locked, not modified

### 2. Predicted Instability (V5.2, new)
- **Source**: `delta_ema` (risk change rate)
- **Scaling**: `delta_ema * 2.0` (gentle, not * 10.0)
- **Cap**: 0.5 (not 1.0)
- **Formula**: `predicted_instability = max(0.0, min(delta_ema * 2.0, 0.5))`

### 3. Gain Modulation (V5.2, new)
- **Type**: Multiplicative (not additive)
- **Formula**: `gain *= (1.0 + 0.2 * predicted_instability)`
- **Max boost**: 10% (when predicted_instability = 0.5)
- **Conservative**: Not +15% or +30%

## Expected Behavior

- **Stable** (delta_ema ≈ 0): No boost
- **Risk increasing** (delta_ema > 0): Mild boost (up to 10%)
- **Max boost**: 10% (when delta_ema = 0.25)

## Validation Criteria

### Good (Stage 0/early Stage 1)
- Mean: **+1% to +3%**
- Std: **±1-2%** (not insane)
- Seed deltas: **+0.5%, +1.2%, +2.1%, +0.8%, +1.7%** (most positive)

### Bad (Still too aggressive)
- Wild mix: **+4% and -5%** (inconsistent)
- High variance: **±3-4%** std
- Mixed signs: Some seeds positive, some very negative

## Testing

Running 5 seeds × 30 episodes validation.

**Target**: +1% to +3% with consistent sign across seeds.

## Next Steps

1. ⏳ Complete V5.2 validation (5 seeds)
2. ⏳ If good (+1% to +3%) → proceed to PREFALL scaling experiments
3. ⏳ If bad (wild mix) → reduce boost multiplier (0.15 instead of 0.2)

