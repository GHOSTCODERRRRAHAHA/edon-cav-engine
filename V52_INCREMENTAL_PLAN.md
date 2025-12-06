# V5.2 Incremental Predicted Boost Testing

## Goal

Find boost configuration that gives +1% to +3% on medium_stress.

## Test Matrix (3 seeds each)

| Test | Boost Mult | Scale | Cap | Max Boost | Formula |
|------|------------|-------|-----|-----------|---------|
| 1 | 0.15 | 1.5 | 0.5 | 7.5% | Very conservative |
| 2 | 0.20 | 2.0 | 0.5 | 10% | Current V5.2 |
| 3 | 0.25 | 2.5 | 0.5 | 12.5% | Slightly more |
| 4 | 0.20 | 1.5 | 0.5 | 7.5% | Lower scale |
| 5 | 0.20 | 2.0 | 0.6 | 12% | Higher cap |

## Process

1. Test each configuration with 3 seeds
2. Compute mean ± std
3. Select best (most positive, in +1% to +3% range)
4. Run full 5-seed validation
5. If target met → proceed to PREFALL scaling

## Expected Results

**Good**:
- Mean: +1% to +3%
- Std: ±1-2%
- Most seeds positive

**Bad**:
- Wild mix of positive/negative
- High variance
- Mean outside target range

## Status

⏳ Running incremental tests (3 seeds each)...
Results will show which boost configuration works best.

