# v8 Ablation Study - Final Summary

## Test Results Comparison

| Test | Interventions/ep | Stability | EDON v8 Score | Verdict |
|------|------------------|-----------|---------------|---------|
| **Baseline** | 40.43 | 0.0206 | 40.73 | Baseline |
| **Reflex-Only** | 0.00 | 0.2669 | ~20 (est) | ❌ REGRESS (stability 13x worse) |
| **Strategy-Only** | 0.00 | 0.2531 | ~20 (est) | ❌ REGRESS (stability 12x worse) |
| **v8 Full** | 40.83 | 0.0222 | 37.31 | ❌ REGRESS |

## Key Findings

### 1. Reflex Controller ❌
- **Interventions**: 0.00 (perfect!)
- **Stability**: 0.2669 (13x worse than baseline 0.0206)
- **Conclusion**: Reflex controller prevents interventions but **destroys stability**
- **Verdict**: **NOT a safe base** - too aggressive, over-dampens

### 2. Strategy Layer ❌
- **Interventions**: 0.00 (perfect!)
- **Stability**: 0.2531 (12x worse than baseline 0.0206)
- **Conclusion**: Strategy layer also prevents interventions but **hurts stability**
- **Verdict**: Strategy layer is also problematic

### 3. Reward Function ✅
- **Correlation**: 0.821 (target: >= 0.7)
- **Conclusion**: Reward function **IS aligned** with EDON v8 score
- **Verdict**: **NOT the problem** - PPO is optimizing correctly

### 4. Fail-Risk Model ⚠️
- **Separation**: 0.1249 (target: >= 0.2)
- **Mean fail-risk (safe)**: 0.5945
- **Mean fail-risk (failure)**: 0.7194
- **Positive rate**: 66% (target: 40-50%)
- **Conclusion**: Fail-risk model has **weak discrimination**
- **Verdict**: **Not a useful control feature** - "always high"

## Root Cause Analysis

### Primary Problem: **Stability Destruction**

Both reflex-only and strategy-only show:
- ✅ **0 interventions** (excellent!)
- ❌ **Stability 12-13x worse** than baseline

**Why?**
1. **Reflex controller over-dampens**: Reduces action magnitude too much, leading to poor stability
2. **Strategy layer learns wrong behavior**: Policy learns to be too conservative
3. **Fail-risk signal is weak**: Poor discrimination (0.12 separation) means it's not useful

### Secondary Problem: **Fail-Risk Model**

- Separation 0.12 < 0.2 threshold
- Mean fail-risk is high for both safe (0.59) and failure (0.72) steps
- "Always high" - not a good control feature

## Recommendations

### 1. **Remove or Drastically Reduce Reflex Controller** ⚠️
- Current reflex controller is too aggressive
- Consider removing it entirely or making it much lighter
- Or only apply reflex in extreme cases (tilt > 0.4)

### 2. **Improve Fail-Risk Model** ⚠️
- Current model has weak discrimination (0.12 separation)
- Need better features or different approach
- Target: separation >= 0.2, positive rate 40-50%

### 3. **Reconsider Architecture** ⚠️
- Layered approach (reflex + strategy) is not working
- Both layers independently hurt stability
- Consider single-layer learned controller (like v7)

### 4. **Reward Function is OK** ✅
- Correlation 0.821 is good
- Not the primary problem
- Keep current reward function

## Next Steps

1. **Test with no reflex controller**: Pure strategy layer
2. **Improve fail-risk model**: Better features, different architecture
3. **Consider single-layer**: Abandon layered approach, use v7-style single layer
4. **Tune strategy layer**: If keeping layered, need to prevent stability destruction

## Conclusion

**The v8 layered architecture is fundamentally flawed:**
- Reflex controller destroys stability
- Strategy layer also hurts stability
- Fail-risk model is not useful
- Reward function is fine (not the problem)

**Recommendation**: **Abandon layered approach** or **drastically simplify it**.

