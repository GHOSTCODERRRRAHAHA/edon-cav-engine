# v8 Ablation Study Results

## 1. Reflex-Only Ablation ✅

**Test**: Baseline + Reflex Controller (no strategy layer)

**Results**:
- Interventions/episode: **0.00** (vs baseline 40.43) ✅ **EXCELLENT**
- Stability (avg): **0.2669** (vs baseline 0.0206) ❌ **MUCH WORSE** (13x worse!)
- Episode length: 306 steps (vs baseline ~333)
- Avg fail-risk: 0.5020

**Verdict**: ⚠️ **Reflex controller is NOT a safe base**
- Prevents interventions but at huge cost to stability
- Too conservative - over-dampens actions
- Stability is 13x worse than baseline

**Conclusion**: Reflex controller alone is too aggressive and hurts stability significantly.

---

## 2. Strategy-Only Ablation

**Test**: Strategy layer with minimal reflex (just clipping/very light damping)

**Status**: Running...

---

## 3. Reward Correlation ✅

**Test**: Correlation between per-episode reward and EDON v8 score

**Results**:
- Correlation: **0.821** ✅ **GOOD** (target: >= 0.7)
- Reward range: [-1518.5, -1009.8]
- EDON v8 Score range: [32.4, 44.1]

**Verdict**: ✅ **Reward function IS aligned with EDON v8 score**
- Correlation 0.821 > 0.7 threshold
- PPO is optimizing what we care about
- Reward function is NOT the primary problem

---

## 4. Fail-Risk Distribution ✅

**Test**: Verify fail-risk model discriminates between safe and failure steps

**Results**:
- Total steps analyzed: 1000
- Mean fail-risk: 0.6662
- **Mean fail-risk (safe steps)**: 0.5945
- **Mean fail-risk (failure steps)**: 0.7194
- **Separation**: 0.1249 ⚠️ **WEAK** (target: >= 0.2)
- Positive rate (fail-risk > 0.5): ~66% (target: 40-50%)

**Verdict**: ⚠️ **Fail-risk discrimination is WEAK**
- Separation 0.1249 < 0.2 threshold
- Mean fail-risk is high for both safe (0.59) and failure (0.72) steps
- Positive rate 66% is outside ideal range (40-50%)
- Fail-risk is "always high" - not a good control feature

**Conclusion**: Fail-risk model needs improvement or different features.

---

## Key Findings

1. **Reflex controller is too aggressive**: Prevents interventions but destroys stability
2. **Reward function is aligned**: Correlation 0.821 is good
3. **Fail-risk model is weak**: Poor discrimination (separation 0.12 < 0.2)
4. **Strategy layer**: Need to test with minimal reflex

## Recommendations

1. **Reduce reflex controller aggressiveness further** or remove it entirely
2. **Improve fail-risk model**: Better features or different approach
3. **Test strategy-only**: See if strategy can learn without aggressive reflex
4. **Consider single-layer architecture**: Maybe layered approach is the problem

