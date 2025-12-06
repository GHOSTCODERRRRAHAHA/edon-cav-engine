# All 3 Recommendations - Implementation Status

## ✅ Recommendation 1: Fix Fail-Risk Model v2

**Status**: FIXED (data loading), but still needs debugging

**Changes Made**:
- Fixed episode loading to use proper `load_episode_from_jsonl()` function
- Fixed feature extraction to properly track history across episodes
- Improved data splitting with proper randomization

**Current Results**:
- ✅ Data loading: Working (18,701 samples from 189 episodes)
- ✅ Feature extraction: Working (19 features per sample)
- ⚠️ Model performance: AUC 0.5, Separation 0.0 (still needs debugging)

**Issue**: Model predictions are all the same (probably predicting majority class)
- Need to check: class imbalance (74.1% positive), feature scaling, model hyperparameters

**Next Steps**:
1. Debug why model isn't learning (check predictions distribution)
2. Try class_weight='balanced' in LightGBM
3. Check feature importance
4. Try different hyperparameters

**Files Modified**:
- `training/fail_risk_model_v2.py`: Fixed data loading and feature extraction

---

## ✅ Recommendation 2: Retrain Strategy Policy Without Reflex

**Status**: TRAINING IN PROGRESS

**Changes Made**:
- Reflex controller already disabled in `env/edon_humanoid_env_v8.py`
- Training started with model name: `edon_v8_strategy_v1_no_reflex`
- Using improved reward function with hidden instability penalties

**Training Command**:
```bash
python training/train_edon_v8_strategy.py \
  --episodes 200 \
  --profile high_stress \
  --seed 0 \
  --lr 5e-4 \
  --gamma 0.995 \
  --update-epochs 10 \
  --output-dir models \
  --model-name edon_v8_strategy_v1_no_reflex \
  --fail-risk-model models/edon_fail_risk_v1_fixed_v2.pt \
  --max-steps 1000
```

**Expected Completion**: ~20-30 minutes

**After Training**:
1. Copy model: `copy models\edon_v8_strategy_v1_no_reflex.pt models\edon_v8_strategy_v1.pt`
2. Test: `python run_eval.py --mode edon --profile high_stress --episodes 30 --seed 42 --output results/v8_retrained_no_reflex.json --edon-gain 1.0 --edon-arch v8_strategy --edon-score`
3. Compare with baseline

**Files Modified**:
- `env/edon_humanoid_env_v8.py`: Reflex controller disabled
- `training/edon_score.py`: Hidden instability penalties added

---

## ✅ Recommendation 3: v7-Style Single-Layer Controller

**Status**: ALREADY SUPPORTED, READY TO USE

**Current State**:
- v7-style controller already exists in `training/train_edon_v7.py`
- Already integrated into `run_eval.py` as `--edon-arch v7_learned`
- Uses single-layer policy that outputs action deltas directly

**Architecture**:
- Input: Observation + baseline action
- Output: Action delta (added to baseline)
- No strategy layer, no reflex layer
- All dynamics visible to one policy

**How to Use**:
1. Train v7 policy (if not already trained):
   ```bash
   python training/train_edon_v7.py --episodes 300 --profile high_stress --seed 0
   ```

2. Evaluate v7:
   ```bash
   python run_eval.py --mode edon --profile high_stress --episodes 30 --seed 42 --output results/v7_improved.json --edon-gain 1.0 --edon-arch v7_learned --edon-score
   ```

3. Compare with baseline:
   ```bash
   python training/compare_v8_vs_baseline.py --baseline results/baseline_v8_final.json --v8 results/v7_improved.json
   ```

**Benefits**:
- Simpler architecture (no layered complexity)
- Uses improved reward function (with hidden instability penalties)
- All control dynamics visible to policy
- No hidden suppression layers

**Files**:
- `training/train_edon_v7.py`: v7 training script
- `run_eval.py`: Already supports `v7_learned` architecture

---

## Summary

| Recommendation | Status | Next Action |
|----------------|--------|--------------|
| 1. Fix Fail-Risk v2 | ⚠️ Fixed loading, needs debugging | Debug model predictions |
| 2. Retrain Strategy | 🔄 Training in progress | Wait for completion, then test |
| 3. v7-Style Single-Layer | ✅ Ready to use | Train/evaluate v7 with improved reward |

---

## Next Steps

1. **Wait for strategy policy training** (~20-30 min)
2. **Test retrained strategy policy** without reflex
3. **Debug fail-risk model v2** (check predictions, try balanced classes)
4. **Test v7-style controller** with improved reward function
5. **Compare all approaches**:
   - Baseline
   - v8 retrained (no reflex)
   - v7-style (single-layer)

---

## Expected Outcomes

- **v8 Retrained**: Should perform better than current v8 (stability should improve)
- **v7-Style**: Should be simpler and potentially more stable
- **Fail-Risk v2**: Once fixed, should provide better discrimination (separation >= 0.25)

