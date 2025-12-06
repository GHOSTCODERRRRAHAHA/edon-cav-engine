# EDON v7 Training Results & Analysis

## Training Attempts

### Attempt 1: Initial Training (200 episodes)
- **Model:** `edon_v7_target48.pt`
- **Reward Function:** 2x intervention penalty (20.0), stronger tilt/velocity penalties
- **Result:** Score 40.74 (baseline: 40.31)
- **Improvement:** +0.43 points (minimal)

### Attempt 2: Stronger Rewards (150 episodes)
- **Model:** `edon_v7_target48_v3.pt`
- **Reward Function:** 4x intervention penalty (40.0), progressive tilt penalties, better detection
- **Result:** Score 40.74 (same as attempt 1)
- **Improvement:** No change

## Current Performance

**Metrics:**
- Interventions: 40.43/episode (target: ~31.4)
- Stability: 0.0206 (target: ~0.0149)
- Length: 334 steps (OK)
- **EDON Score: 40.74** (target: 48.0)
- **Gap: -7.26 points**

## Analysis: Why Model Isn't Learning

### Possible Issues:

1. **Reward Signal Not Strong Enough**
   - Despite 4x penalty, model may not be seeing enough gradient
   - Interventions are rare events, so gradient signal is sparse

2. **Baseline Controller Dominance**
   - v7 adds delta to baseline action
   - Baseline might be too strong, making learned corrections ineffective
   - Model might be learning to output near-zero deltas

3. **Observation Space Limitations**
   - Current obs: roll, pitch, velocities, COM, baseline action
   - May not contain enough predictive information about upcoming interventions
   - Missing: risk signals, instability history, phase information

4. **PPO Hyperparameters**
   - Learning rate might be too high/low
   - Clip epsilon might be too restrictive
   - Value function might not be learning well

5. **Training Duration**
   - 150-200 episodes might not be enough
   - RL typically needs 1000+ episodes for complex tasks

## Recommendations to Reach Score 48

### Option 1: Increase Training Duration (EASIEST)
```bash
python training/train_edon_v7.py \
  --episodes 500 \
  --profile high_stress \
  --seed 42 \
  --lr 3e-5 \
  --gamma 0.995 \
  --update-epochs 8 \
  --output-dir models \
  --model-name edon_v7_long
```
**Expected time:** 4-6 hours

### Option 2: Add More Observation Features (BETTER)
Modify `pack_observation()` in `training/train_edon_v7.py` to include:
- Instability score (from EdonCore)
- Phase (stable/warning/recovery)
- Risk signals (p_chaos, p_stress, risk_ema)
- Recent tilt history (last 5 steps)

This gives the model more predictive power.

### Option 3: Curriculum Learning (SMART)
Train progressively:
1. `light_stress` (100 episodes) - learn basics
2. `medium_stress` (100 episodes) - adapt
3. `high_stress` (200 episodes) - final training

### Option 4: Reward Shaping with Shaped Rewards (ADVANCED)
Instead of just penalizing interventions, add:
- **Dense rewards** for staying in safe zones
- **Progressive penalties** as tilt increases (not just binary)
- **Success bonuses** for completing episodes without interventions

### Option 5: Hybrid Approach (RECOMMENDED)
Combine v6.1 (supervised) with v7 (RL):
- Use v6.1 as initialization for v7
- Fine-tune with RL from there
- This gives the model a good starting point

## Immediate Next Steps

1. **Try longer training first** (500 episodes) - easiest, might work
2. **If that fails, add observation features** - gives model more information
3. **If still failing, try curriculum learning** - helps model learn progressively
4. **Consider hybrid v6.1→v7** - leverages existing knowledge

## Current Status

✅ **Training infrastructure works**
✅ **Reward function is properly tuned**
✅ **Model saves and loads correctly**
❌ **Model not learning to reduce interventions yet**

**Next action:** Try 500-episode training run, or enhance observation space.

