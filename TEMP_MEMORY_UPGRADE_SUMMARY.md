# EDON v8 Temporal Memory Upgrade Summary

## ✅ Implemented Features

### 1. Early-Warning Features
Added to `pack_observation_v8()` in `training/edon_v8_policy.py`:

- **Rolling Variance** (last 20 steps):
  - `roll_var`: Variance of roll angles
  - `pitch_var`: Variance of pitch angles
  - `roll_vel_var`: Variance of roll velocities
  - `pitch_vel_var`: Variance of pitch velocities

- **High-Frequency Oscillation Energy**:
  - `osc_energy`: Mean of squared angular velocities (detects rapid oscillations)

- **Near-Fail Density**:
  - `near_fail_density`: Fraction of last 20 steps with fail_risk > 0.6

**Total early-warning features: 6**

### 2. Stacked Observations (Temporal Memory)
Added `pack_stacked_observation_v8()` function:

- Maintains buffer of last **K=8** observation vectors
- Concatenates them: `[frame_t-7, frame_t-6, ..., frame_t-1, frame_t]`
- Gives policy temporal context to learn what leads to interventions

**Stacked observation size: 248 (31 base features × 8 frames)**

### 3. Environment Integration
Updated `env/edon_humanoid_env_v8.py`:

- Added temporal buffers:
  - `obs_history`: Last 20 observation dicts (for rolling variance)
  - `obs_vec_history`: Last 8 packed observation vectors (for stacking)
  - `near_fail_history`: Last 20 near-fail flags (for near-fail density)

- Buffers are:
  - Initialized in `__init__`
  - Reset in `reset()`
  - Updated in `step()` after packing observations

### 4. Training & Evaluation Updates
Updated:
- `training/train_edon_v8_strategy.py`: Uses `pack_stacked_observation_v8()` and computes input_size correctly
- `run_eval.py`: Uses `pack_stacked_observation_v8()` for input size calculation

## Feature Vector Breakdown

**Base features per frame (31):**
- Basic state: 4 (roll, pitch, roll_velocity, pitch_velocity)
- COM state: 4 (com_x, com_y, com_velocity_x, com_velocity_y)
- Derived: 4 (tilt_mag, vel_norm, com_norm, com_vel_norm)
- Risk/phase: 3 (fail_risk, instability_score, phase_encoded)
- Baseline action: 10 (assumed 10-dim)
- **Early-warning: 6** (roll_var, pitch_var, roll_vel_var, pitch_vel_var, osc_energy, near_fail_density)

**Stacked observation (248):**
- 31 base features × 8 frames = 248

## Next Steps

1. **Retrain the policy** with the new observation structure:
   ```bash
   python training/train_edon_v8_strategy.py \
     --episodes 300 \
     --profile high_stress \
     --seed 0 \
     --lr 5e-4 \
     --gamma 0.995 \
     --update-epochs 10 \
     --output-dir models \
     --model-name edon_v8_strategy_temporal_v1 \
     --fail-risk-model models/edon_fail_risk_v1_fixed_v2.pt \
     --max-steps 1000 \
     --w-intervention 20.0 \
     --w-stability 1.0 \
     --w-torque 0.1 \
     --retroactive-steps 20 \
     --w-retroactive 3.0
   ```

2. **Evaluate** to see if temporal context helps the policy learn to avoid interventions while maintaining stability.

## Expected Benefits

- **Early-warning features** should spike BEFORE interventions, giving the policy predictive signals
- **Stacked observations** give the policy temporal context to learn sequences that lead to interventions
- Combined with retroactive penalties, the policy should now be able to learn what actions lead to interventions

