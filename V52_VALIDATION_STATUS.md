# V5.2 Validation Status

## Configuration

**V5.2 = V5.1 (LPF) + Super-Conservative Predicted Boost**

- **LPF**: `alpha = 0.75 - 0.15 * instability` (unchanged from V5.1)
- **Predicted instability**: `delta_ema * 2.0`, capped at 0.5
- **Boost**: `gain *= (1.0 + 0.2 * predicted_instability)` (up to +10%)

## Initial Test (Seed 1)

- **Interventions**: +0.2%
- **Stability**: -0.1%
- **Average**: **+0.0%** (neutral)

**Status**: Better than negative, but below +1% to +3% target.

## Full Validation

Running 5 seeds × 30 episodes to check:
- **Consistency**: Are most seeds positive?
- **Variance**: Is std reasonable (±1-2%)?
- **Mean**: Does it reach +1% to +3%?

## Expected Outcomes

### Good (Stage 0/early Stage 1)
- Mean: +1% to +3%
- Std: ±1-2%
- Seed deltas: +0.5%, +1.2%, +2.1%, +0.8%, +1.7% (most positive)

### Bad (Still too aggressive or misaligned)
- Wild mix: +4% and -5%
- High variance: ±3-4%
- Mixed signs

## Next Steps

1. ⏳ Complete 5-seed validation
2. ⏳ If mean is +1% to +3% → proceed to PREFALL scaling
3. ⏳ If mean is 0% to +1% → may need slightly more boost (0.25 instead of 0.2)
4. ⏳ If wild mix → reduce boost further (0.15 instead of 0.2)

