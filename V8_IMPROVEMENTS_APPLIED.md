# v8 Performance Improvements Applied

## Issues Identified

1. **Fail-risk model too aggressive**: 98% positive labels (marking "prefall" as failure)
2. **Suboptimal advantage computation**: Simple mean baseline instead of GAE
3. **Missing entropy bonus**: No exploration incentive in loss function
4. **Wrong fail-risk model in evaluation**: Using old model instead of fixed version

## Fixes Applied

### 1. Fixed Fail-Risk Label Detection ✅

**Problem**: Label detection was marking "prefall" as failure, causing 98% positive labels.

**Fix**: Updated `compute_fail_label()` in `training/fail_risk_model.py`:
- Only mark actual "fail" zone, not "prefall"
- Increased extreme tilt threshold from 0.35 to 0.40
- More selective failure detection

**Results**:
- Before: 98.0% positive labels
- After: 75.9% positive labels (more balanced)
- Validation AUC: 0.8666 (good discrimination)
- Validation accuracy: 81.13%

### 2. Improved PPO Training ✅

**Problem**: Simple advantage computation (`returns - mean_return`) is suboptimal.

**Fix**: Added GAE (Generalized Advantage Estimation) to `training/train_edon_v8_strategy.py`:
- Added `compute_gae()` method with `gae_lambda=0.95`
- Uses returns as value estimates (simpler than learning separate value function)
- Better advantage estimation for policy updates

**Also added**:
- Entropy coefficient (`entropy_coef=0.01`) for exploration
- Entropy bonus in loss function

### 3. Updated Model Paths ✅

**Fix**: Updated `run_eval.py` to use fixed fail-risk model:
- Tries `edon_fail_risk_v1_fixed_v2.pt` first (newest)
- Falls back to `edon_fail_risk_v1_fixed.pt`
- Falls back to `edon_fail_risk_v1.pt` (original)

**Fix**: Updated default fail-risk model path in training script.

## Expected Improvements

With these fixes, the retrained v8 policy should:

1. **Better fail-risk predictions**: More balanced (75.9% vs 98%), better AUC (0.87)
2. **Better learning**: GAE provides better advantage estimates
3. **Better exploration**: Entropy bonus encourages strategy diversity
4. **Better performance**: Should see improvement over baseline

## Training Status

**Currently training**: v8 strategy policy with improved setup
- Model: `models/edon_v8_strategy_v1_improved.pt`
- Fail-risk model: `models/edon_fail_risk_v1_fixed_v2.pt`
- Episodes: 300
- Improvements: GAE, entropy bonus, better fail-risk model

## Next Steps

After training completes:

1. **Evaluate baseline** (if not already done):
```bash
python run_eval.py --mode baseline --profile high_stress --episodes 30 --seed 42 --output results/baseline_v8_final.json --edon-score
```

2. **Evaluate improved v8**:
```bash
python run_eval.py --mode edon --profile high_stress --episodes 30 --seed 42 --output results/edon_v8_improved_final.json --edon-gain 1.0 --edon-arch v8_strategy --edon-score
```

3. **Compare**:
```bash
python training/compare_v8_vs_baseline.py --baseline results/baseline_v8_final.json --v8 results/edon_v8_improved_final.json
```

**Expected**: [PASS] with 5-15% improvement over baseline

