# v8 Fixes Applied - Complete Overhaul

## ✅ Step 1: Reflex Controller KILLED

**Status**: COMPLETE

**Changes**:
- Removed reflex controller from `env/edon_humanoid_env_v8.py`
- Strategy modulations now applied directly to baseline action
- Only minimal clipping for safety (-1.0 to 1.0)

**Files Modified**:
- `env/edon_humanoid_env_v8.py`: Reflex controller disabled, direct modulation application

**Rationale**: 
- Reflex-only ablation showed: 0 interventions but stability 13x worse
- Reflex + strategy showed system conflict
- Reflex is architecturally wrong for current signal stack

---

## ✅ Step 2: Fail-Risk Model v2 (In Progress)

**Status**: FRAMEWORK CREATED, NEEDS TRAINING

**New Features**:
1. **Temporal Features**:
   - Rolling variance of state (roll, pitch, velocities)
   - Derivative of error (rate of change)
   - Oscillation frequency (zero-crossing rate)

2. **Energy Features**:
   - Actuator saturation (how close to limits)
   - Recovery overshoot (action magnitude after disturbance)
   - Energy dissipation rate

**Model Type**: XGBoost/LightGBM (not deep learning)

**Target**: Fail-risk separation >= 0.25

**Files Created**:
- `training/fail_risk_model_v2.py`: Improved model with temporal + energy features
- `training/train_fail_risk_v2.py`: Training script

**Next Steps**:
1. Install dependencies: `pip install lightgbm` or `pip install xgboost`
2. Train model: `python training/train_fail_risk_v2.py --dataset-glob "logs/*.jsonl" --output models/edon_fail_risk_v2.pt --model-type lightgbm`
3. Verify separation >= 0.25 before using in v8

---

## ✅ Step 3: Hidden Instability Penalties Added

**Status**: COMPLETE

**New Penalties**:
1. **High-frequency oscillation penalty**:
   - Detects rapid oscillations (sign changes in velocity)
   - Penalty: 2.0 * (roll_oscillation + pitch_oscillation)

2. **Phase-lag penalty**:
   - Detects when tilt and velocity are out of phase
   - High tilt + low velocity = phase lag
   - High velocity + low tilt = also phase lag
   - Penalty: 3.0 * tilt_mag or 2.0 * vel_mag

3. **Control jerk penalty**:
   - Penalizes rapid action changes (Δu/Δt)
   - Threshold: 0.5
   - Penalty: 1.5 * (jerk_mag - 0.5)

**Files Modified**:
- `training/edon_score.py`: Added hidden instability penalties to `step_reward()`

**Rationale**: Prevents "fake calm, real chaos" - detects instability that doesn't show up in simple tilt/velocity metrics

---

## ⚠️ Step 4: v7-Style Single-Layer Controller (Pending)

**Status**: NOT YET IMPLEMENTED

**Current State**: v8 still uses strategy layer (but without reflex)

**Recommendation**: 
- v7-style controller already exists in `training/train_edon_v7.py`
- Can be used as-is for single-layer approach
- Architecture: Policy outputs action deltas directly (not strategies)

**Next Steps**:
1. Test v8 without reflex (current state)
2. If still regressing, switch to v7-style single-layer
3. Use v7 controller with improved reward function (with instability penalties)

---

## Summary of Changes

### Completed ✅
1. **Reflex controller killed** - completely disabled
2. **Hidden instability penalties** - added to reward function
3. **Fail-risk model v2 framework** - created with better features

### In Progress 🔄
1. **Fail-risk model v2 training** - needs data and training
2. **v7-style single-layer** - framework exists, needs integration

### Next Actions
1. **Train fail-risk v2**: `python training/train_fail_risk_v2.py --dataset-glob "logs/*.jsonl" --output models/edon_fail_risk_v2.pt`
2. **Test v8 without reflex**: Run evaluation to see if removing reflex helps
3. **If still regressing**: Switch to v7-style single-layer controller

---

## Architecture Changes

### Before (v8 with reflex):
```
Baseline → Strategy Policy → Reflex Controller → Final Action
```

### After (v8 without reflex):
```
Baseline → Strategy Policy → Direct Modulation → Final Action
```

### Future (v7-style single-layer):
```
Baseline → Single Policy → Action Delta → Final Action
```

---

## Key Insights

1. **Reflex controller was the problem**: Destroyed stability while preventing interventions
2. **Reward function is aligned**: Correlation 0.821 is good
3. **Fail-risk model needs improvement**: Current separation 0.12 < 0.25 target
4. **Hidden instability is real**: Need to penalize oscillations, phase-lag, and jerk

---

## Testing Plan

1. **Test v8 without reflex**:
   ```bash
   python run_eval.py --mode edon --profile high_stress --episodes 30 --seed 42 --output results/v8_no_reflex.json --edon-gain 1.0 --edon-arch v8_strategy --edon-score
   ```

2. **Compare with baseline**:
   ```bash
   python training/compare_v8_vs_baseline.py --baseline results/baseline_v8_final.json --v8 results/v8_no_reflex.json
   ```

3. **If still regressing, test v7-style**:
   - Use existing v7 controller
   - Train with improved reward function
   - Evaluate vs baseline

