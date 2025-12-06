# LPF Incremental Testing

## Goal
Find alpha formula that gives +1% to +3% improvement (consistent sign).

## Test Matrix

| Test | Formula | Base | Scale | Range |
|------|---------|------|-------|-------|
| 1 | 0.75 - 0.15 * instability | 0.75 | 0.15 | 75% → 60% |
| 2 | 0.80 - 0.20 * instability | 0.80 | 0.20 | 80% → 60% |
| 3 | 0.70 - 0.10 * instability | 0.70 | 0.10 | 70% → 60% |
| 4 | 0.85 - 0.25 * instability | 0.85 | 0.25 | 85% → 60% |
| 5 | 0.72 - 0.12 * instability | 0.72 | 0.12 | 72% → 60% |

## Testing Process

1. Test each alpha formula with 3 seeds × 30 episodes
2. Compute mean ± std
3. Find formula with best (most positive) mean
4. If best is positive → use it for V5.1
5. If all negative → try even lower alpha values

## Expected Results

- **V4 (no LPF)**: -1.7% ± 1.6%
- **V5.1 initial (0.85-0.3)**: -3.3% ± 2.4% (too much smoothing)
- **V5.1a (0.8-0.2)**: [Testing...]
- **Target**: +1% to +3% (consistent sign)

## Next Steps

After finding best alpha:
1. Lock V5.1 with best alpha formula
2. Run full 5-seed validation
3. If positive → proceed to V5.2 (add mild predicted boost)
4. If still negative → try different LPF approach

