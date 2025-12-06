# v8 Performance Fixes - Round 2

## Issues Found

1. **Reflex controller too aggressive**: Multiplicative damping (1.5^3 potential) was reducing actions too much
2. **Reward function**: Alive bonus too low (0.6), not incentivizing longer episodes enough
3. **Exploration insufficient**: Entropy coefficient too low (0.01)

## Fixes Applied

### 1. Reduced Reflex Controller Aggressiveness ✅

**Changes:**
- `max_damping_factor`: 1.5 → **1.3** (allows more action)
- `fail_risk_damping_scale`: 2.0 → **1.5** (less aggressive fail-risk response)
- Damping combination: **Multiplicative → Max** (prevents compounding)

**Before**: `total_damping = min(1.5, tilt * vel * fail_risk)` could compound
**After**: `total_damping = min(1.3, max(tilt, vel, fail_risk))` uses max, capped at 1.3

**Impact**: Actions are now reduced to at most 77% (1/1.3) instead of 67% (1/1.5), and modulations can have more effect.

### 2. Improved Reward Function ✅

**Changes:**
- Alive bonus: 0.6 → **0.8** per step

**Impact**: Better incentive for longer episodes, better alignment with EDON score length_bonus.

### 3. Increased Exploration ✅

**Changes:**
- Entropy coefficient: 0.01 → **0.02** (doubled)

**Impact**: Policy will explore more strategies, less likely to converge to suboptimal local minimum.

## Training Status

**Currently training**: v8 strategy policy with all fixes
- Model: `models/edon_v8_strategy_v1_fixed.pt`
- Episodes: 200 (faster iteration)
- Improvements:
  - Less aggressive reflex controller
  - Better reward alignment
  - More exploration

## Expected Improvements

With these fixes:
1. **More action authority**: Reflex controller won't over-dampen
2. **Better learning signal**: Reward function better aligned with EDON score
3. **More exploration**: Policy will try more strategies

**Expected**: 5-10% improvement over baseline (interventions down, stability similar or better)

## Next Steps

After training completes (~20-30 minutes):
1. Evaluate: `python run_eval.py --mode edon --profile high_stress --episodes 30 --seed 42 --output results/edon_v8_fixed_final.json --edon-gain 1.0 --edon-arch v8_strategy --edon-score`
2. Compare: `python training/compare_v8_vs_baseline.py --baseline results/baseline_v8_final.json --v8 results/edon_v8_fixed_final.json`

