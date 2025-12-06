# V5.2 Configuration - Predicted Instability Boost

## Changes from V5.1

**V5.1**: LPF only (alpha = 0.75 - 0.15 * instability)
**V5.2**: V5.1 + super-conservative predicted instability boost

## Implementation

### 1. Predicted Instability Calculation
```python
# Super-conservative: delta_ema * 2.0, capped at 0.5 (not too spicy)
predicted_instability = max(0.0, min(delta_ema * 2.0, 0.5))
```

**Key points**:
- Uses `delta_ema` as proxy (not delta_ema * 10.0)
- Gentle scaling: * 2.0 (not * 10.0)
- Capped at 0.5 (not 1.0)

### 2. Small Gain Modulation
```python
# Mild predicted boost: up to +10% (not +15% or +30%)
gain *= (1.0 + 0.2 * predicted_instability)  # 0-10% bump
```

**Key points**:
- Multiplicative (not additive)
- 0.2 * 0.5 max = 0.1 = 10% boost
- Small modulation, not a turbo button

### 3. LPF Unchanged
- **Keep exactly as V5.1**: `alpha = 0.75 - 0.15 * instability_score`
- Want to measure incremental effect of prediction, not filter changes

## Expected Results

### Good (Stage 0/early Stage 1)
- **Mean**: +1% to +3%
- **Std**: ±1-2% (not insane)
- **Seed deltas**: +0.5%, +1.2%, +2.1%, +0.8%, +1.7% (most positive)

### Bad (Still too aggressive)
- **Wild mix**: +4% and -5% (inconsistent)
- **High variance**: ±3-4% std
- **Mixed signs**: Some seeds positive, some very negative

## Testing

Running 5 seeds × 30 episodes validation.

**Target**: +1% to +3% with consistent sign across seeds.

## Next Steps

1. ⏳ Complete V5.2 validation (5 seeds)
2. ⏳ If good (+1% to +3%) → proceed to PREFALL scaling experiments
3. ⏳ If bad (wild mix) → reduce predicted boost further (0.15 instead of 0.2)

