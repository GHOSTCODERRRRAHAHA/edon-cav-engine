# Concrete Action Plan to Get to 20%

## Current Status
- **Best**: V5.1 (LPF only) = +0.2%
- **Target**: 20%
- **Gap**: 19.8 percentage points

## Phase-by-Phase Plan

### Phase 1: Get to +1% to +3% (Stage 0) - THIS WEEK

#### Step 1: Remove Predicted Boost (Immediate)
**Problem**: V5.2 with predicted boost = -1.4% (hurting)
**Action**: 
- Remove predicted boost code from `apply_edon_gain()` in `evaluation/edon_controller_v3.py`
- Go back to V5.1 (LPF only) which gives +0.2%
- **File**: `evaluation/edon_controller_v3.py`, line ~60
- **Change**: Remove `gain *= (1.0 + 0.2 * predicted_instability)`

#### Step 2: Test LPF on Medium Stress
**Why**: V5.1 was tested on high_stress, need to verify on medium
**Action**:
```bash
python run_eval.py --mode edon --episodes 30 --profile medium_stress --edon-gain 0.75 --output results/v5.1_medium_stress.json
python run_eval.py --mode baseline --episodes 30 --profile medium_stress --output results/baseline_medium_stress.json
python plot_results.py results/baseline_medium_stress.json results/v5.1_medium_stress.json
```

#### Step 3: Optimize PREFALL Range
**Current**: PREFALL_MIN=0.15, PREFALL_RANGE=0.50, PREFALL_MAX=0.70
**Action**: Test different ranges incrementally
- Test 1: 0.10-0.50 (more conservative)
- Test 2: 0.12-0.55 (slightly more)
- Test 3: 0.18-0.65 (more aggressive)
- **File**: `evaluation/edon_controller_v3.py`, line ~167-169

#### Step 4: Fine-tune LPF Alpha
**Current**: `alpha = 0.75 - 0.15 * instability`
**Action**: Test variations
- Test 1: `0.73 - 0.15 * instability` (less smoothing)
- Test 2: `0.77 - 0.15 * instability` (more smoothing)
- Test 3: `0.75 - 0.13 * instability` (less aggressive scaling)
- Test 4: `0.75 - 0.17 * instability` (more aggressive scaling)
- **File**: Need to find where LPF alpha is computed

#### Step 5: Verify EDON State Mapping
**Action**: 
- Check what EDON actually outputs
- Verify `map_edon_output_to_state()` is correct
- Ensure we're using right signals
- **File**: `run_eval.py`, search for `map_edon_output_to_state`

**Expected Result**: +1% to +3% after these 5 steps

---

### Phase 2: Get to +5% to +10% (Stage 1) - NEXT 1-2 MONTHS

#### Step 1: Multi-Layer Corrections
**Current**: Simple PD on tilt
**Action**: Add velocity and acceleration corrections
- Tilt correction (current)
- Velocity correction (new)
- Acceleration correction (new)
- **File**: `run_eval.py`, `apply_edon_regulation()`

#### Step 2: Joint-Specific Corrections
**Current**: Global corrections
**Action**: Different corrections for different joints
- Balance joints (torso/hip/ankle): stronger corrections
- Other joints: lighter corrections
- **File**: `run_eval.py`, `apply_edon_regulation()`

#### Step 3: Adaptive PD Gains
**Current**: Fixed PD gains
**Action**: Gains that adapt based on state
- Higher gains in PREFALL
- Lower gains in SAFE
- **File**: `run_eval.py`, `apply_edon_regulation()`

#### Step 4: Better State Transitions
**Current**: Basic state mapping
**Action**: Smooth state transitions
- Hysteresis to prevent oscillation
- Gradual transitions
- **File**: `run_eval.py`, state mapping code

**Expected Result**: +5% to +10% after these improvements

---

### Phase 3: Get to +10% to +15% (Stage 2) - NEXT 3-6 MONTHS

#### Step 1: Model Predictive Control (MPC)
**Action**: Integrate MPC for look-ahead control
- Predict future states
- Optimize trajectory
- **New file**: `evaluation/mpc_controller.py`

#### Step 2: Hierarchical Control
**Action**: Multi-layer control architecture
- High-level: trajectory planning
- Low-level: joint control
- **New file**: `evaluation/hierarchical_controller.py`

#### Step 3: Disturbance Estimation
**Action**: Estimate and reject disturbances
- Kalman filter for disturbance estimation
- Feedforward compensation
- **New file**: `evaluation/disturbance_estimator.py`

#### Step 4: Predictive Control
**Action**: Anticipatory corrections
- Look-ahead prediction
- Pre-emptive corrections
- **File**: `run_eval.py`, add prediction layer

**Expected Result**: +10% to +15% after these changes

---

### Phase 4: Get to +15% to +20% (Stage 3) - NEXT 6-12 MONTHS

#### Step 1: Fundamental Architecture Redesign
**Action**: Redesign control architecture
- Better state estimation
- Advanced filtering
- **Major refactor**: Multiple files

#### Step 2: Machine Learning Integration
**Action**: Learned correction policies
- Train on successful corrections
- Adaptive parameter tuning
- **New file**: `evaluation/ml_controller.py`

#### Step 3: Optimal Control Integration
**Action**: LQR/LQG controllers
- Optimal trajectory tracking
- Constrained optimization
- **New file**: `evaluation/optimal_controller.py`

#### Step 4: Advanced State Estimation
**Action**: Kalman filtering and sensor fusion
- Better uncertainty handling
- Multi-sensor fusion
- **New file**: `evaluation/state_estimator.py`

**Expected Result**: +15% to +20% after these fundamental changes

---

## Immediate Next Steps (This Week)

### Priority 1: Remove Predicted Boost
```python
# In evaluation/edon_controller_v3.py, line ~60
# REMOVE THIS LINE:
gain *= (1.0 + 0.2 * predicted_instability)  # V5.2 predicted boost
```

### Priority 2: Test LPF on Medium Stress
```bash
# Run V5.1 on medium_stress
python run_eval.py --mode edon --episodes 30 --profile medium_stress --edon-gain 0.75 --output results/v5.1_medium.json
python run_eval.py --mode baseline --episodes 30 --profile medium_stress --output results/baseline_medium.json
python plot_results.py results/baseline_medium.json results/v5.1_medium.json
```

### Priority 3: Optimize PREFALL
```python
# In evaluation/edon_controller_v3.py, test different ranges:
# Test 1: More conservative
"PREFALL_MIN": 0.10,
"PREFALL_RANGE": 0.40,
"PREFALL_MAX": 0.50,

# Test 2: Current
"PREFALL_MIN": 0.15,
"PREFALL_RANGE": 0.50,
"PREFALL_MAX": 0.70,

# Test 3: More aggressive
"PREFALL_MIN": 0.18,
"PREFALL_RANGE": 0.47,
"PREFALL_MAX": 0.65,
```

### Priority 4: Fine-tune LPF Alpha
```python
# Find where LPF alpha is computed and test variations
# Current: alpha = 0.75 - 0.15 * instability
# Test: alpha = 0.73 - 0.15 * instability (less smoothing)
# Test: alpha = 0.77 - 0.15 * instability (more smoothing)
```

### Priority 5: Verify State Mapping
```python
# In run_eval.py, check map_edon_output_to_state()
# Verify it's using correct EDON outputs
# Ensure state transitions are correct
```

## Success Criteria

- **Stage 0** (+1% to +3%): Achieve consistent +1% improvement
- **Stage 1** (+5% to +10%): Achieve consistent +5% improvement
- **Stage 2** (+10% to +15%): Achieve consistent +10% improvement
- **Stage 3** (+15% to +20%): Achieve consistent +15% improvement

## Key Principle

**Incremental, systematic improvements.**
- Don't try to jump to 20% directly
- Get to Stage 0 first (+1% to +3%)
- Then build from there
- Each stage builds on the previous

