# V5.1 Incremental LPF Testing - Status

## Current Implementation

**LPF Alpha Formula**: `0.75 - 0.15 * instability_score`
- Stable (instability=0): 75% smoothing
- Unstable (instability=1): 60% smoothing
- Clamped: 0.4 to 0.9

## Testing Strategy

1. **Incremental alpha testing**: Testing 5 different formulas with 3 seeds each
2. **Select best**: Choose formula with most positive mean
3. **Full validation**: Run 5 seeds with best formula
4. **Proceed**: If positive → V5.2 (add predicted boost)

## Test Matrix

| Test | Formula | Status |
|------|---------|--------|
| 1 | 0.75 - 0.15 * instability | [Testing...] |
| 2 | 0.80 - 0.20 * instability | [Testing...] |
| 3 | 0.70 - 0.10 * instability | [Testing...] |
| 4 | 0.85 - 0.25 * instability | [Testing...] |
| 5 | 0.72 - 0.12 * instability | [Testing...] |

## Expected Outcome

- **Target**: +1% to +3% (consistent sign)
- **If all negative**: Try even lower alpha (0.65-0.70 base)
- **If positive**: Lock formula and proceed to V5.2

## Next Steps

1. ⏳ Wait for incremental tests to complete
2. ⏳ Select best alpha formula
3. ⏳ Run full 5-seed validation
4. ⏳ If positive → proceed to V5.2

