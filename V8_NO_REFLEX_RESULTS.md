# v8 Without Reflex - Test Results

## Test Configuration

- **Architecture**: v8 strategy layer (reflex controller DISABLED)
- **Fail-risk model**: v1_fixed_v2 (existing model)
- **Episodes**: 30
- **Profile**: high_stress
- **Seed**: 42

## Results

### v8 Without Reflex:
- **Interventions/episode**: 40.30
- **Stability (avg)**: 0.0211
- **Episode length (avg)**: 320.3 steps
- **EDON v8 Score**: 38.06

### Baseline (for comparison):
- **Interventions/episode**: 40.43
- **Stability (avg)**: 0.0206
- **Episode length (avg)**: ~333 steps
- **EDON v8 Score**: 40.73

## Comparison

| Metric | Baseline | v8 No Reflex | Delta |
|--------|----------|--------------|-------|
| Interventions/ep | 40.43 | 40.30 | -0.3% ✅ (slightly better) |
| Stability | 0.0206 | 0.0211 | +2.4% ⚠️ (slightly worse) |
| EDON v8 Score | 40.73 | 38.06 | -2.67 ❌ (regression) |

## Analysis

### ✅ Improvements:
- **Interventions**: Slightly reduced (-0.3%)
- **Stability**: Only slightly worse (+2.4%, within noise)

### ❌ Still Regressing:
- **EDON v8 Score**: -2.67 points (still regressing)
- **Episode length**: Shorter (320 vs 333 steps)

## Key Findings

1. **Removing reflex helped stability**: 
   - Previous v8 with reflex: stability 0.0222 (worse)
   - v8 without reflex: stability 0.0211 (better, closer to baseline)

2. **But still not better than baseline**:
   - EDON v8 Score still lower than baseline
   - Strategy layer alone is not sufficient

3. **Fail-risk model v2 training failed**:
   - AUC: 0.5 (random)
   - Separation: 0.0
   - Need to fix feature extraction/data loading

## Recommendations

1. **Fix fail-risk model v2**: 
   - Debug feature extraction
   - Fix data loading from JSONL
   - Target: separation >= 0.25

2. **Consider v7-style single-layer**:
   - Strategy layer alone is not working
   - v7-style direct action deltas might work better
   - Use existing v7 controller with improved reward

3. **Retrain strategy policy**:
   - Current policy was trained with reflex
   - Retrain without reflex for better performance

## Next Steps

1. Fix fail-risk model v2 training
2. Retrain v8 strategy policy without reflex
3. Test v7-style single-layer controller
4. Compare all approaches

