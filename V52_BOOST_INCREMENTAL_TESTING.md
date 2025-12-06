# V5.2 Predicted Boost Incremental Testing

## Test Matrix

Testing different combinations of:
- **Boost multiplier**: 0.15, 0.20, 0.25 (controls max boost %)
- **Scale factor**: 1.5, 2.0, 2.5 (controls delta_ema scaling)
- **Cap**: 0.5, 0.6 (controls max predicted_instability)

| Test | Boost Mult | Scale | Cap | Max Boost | Status |
|------|------------|-------|-----|----------|--------|
| 1 | 0.15 | 1.5 | 0.5 | 7.5% | ⏳ |
| 2 | 0.20 | 2.0 | 0.5 | 10% | ⏳ (Current V5.2) |
| 3 | 0.25 | 2.5 | 0.5 | 12.5% | ⏳ |
| 4 | 0.20 | 1.5 | 0.5 | 7.5% | ⏳ |
| 5 | 0.20 | 2.0 | 0.6 | 12% | ⏳ |

## Formulas

**Predicted instability**:
```python
predicted_instability = max(0.0, min(delta_ema * scale, cap))
```

**Gain modulation**:
```python
gain *= (1.0 + boost_mult * predicted_instability)
```

**Max boost calculation**:
- Max boost = `boost_mult * cap`
- Test 1: 0.15 * 0.5 = 7.5%
- Test 2: 0.20 * 0.5 = 10% (current)
- Test 3: 0.25 * 0.5 = 12.5%
- Test 5: 0.20 * 0.6 = 12%

## Testing

- **Profile**: medium_stress
- **Seeds**: 3 per test (faster iteration)
- **Episodes**: 30 per seed

## Target

**Good**: +1% to +3% mean, ±1-2% std, most seeds positive

## Next Steps

1. ⏳ Complete incremental tests
2. ⏳ Select best configuration
3. ⏳ Run full 5-seed validation with best config
4. ⏳ If target met → proceed to PREFALL scaling

