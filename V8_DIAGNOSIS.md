# v8 Performance Diagnosis

## Current Results

**Baseline:**
- Interventions/ep: 40.43
- Stability: 0.0206
- EDON Score: 40.73

**v8 (300 episodes trained):**
- Interventions/ep: 40.50 (+0.2%)
- Stability: 0.0216 (+4.5% worse)
- EDON v8 Score: 37.77 (-2.97 points)

**Verdict: [REGRESS]**

## Issues Identified

### 1. Reward Function Alignment
The reward function penalizes:
- Interventions: -25.0 per intervention
- Tilt: -8.0 * tilt_mag (base) + extra for dangerous tilt
- Velocity: -3.0 * vel_mag
- Action deltas: -0.3 * ||delta||
- Alive bonus: +0.6 per step

**Problem**: The policy might be learning to be too conservative (high damping) to avoid penalties, which reduces effectiveness.

### 2. Reflex Controller Dominance
The reflex controller:
1. Applies damping FIRST (divides action by damping factor)
2. Then applies strategy modulations (gain_scale, lateral_compliance)

**Problem**: If damping is high (e.g., 1.5x), actions are reduced to 67% before modulations. The policy might not be able to overcome this.

### 3. Fail-Risk Model Impact
Even with improved model (75.9% positives, AUC 0.87), the fail-risk signal might:
- Be too noisy
- Not provide useful predictive signal
- Cause over-conservative behavior

### 4. Strategy Policy Learning
The policy might be:
- Not exploring enough (entropy too low)
- Converging to a suboptimal strategy
- Not effectively using modulations

## Potential Fixes

### Option 1: Reduce Reflex Controller Aggressiveness
- Lower `max_damping_factor` from 1.5 to 1.2
- Reduce `fail_risk_damping_scale` from 2.0 to 1.5
- Allow strategy modulations to have more effect

### Option 2: Improve Reward Function
- Increase alive bonus to encourage longer episodes
- Reduce tilt penalty to allow more exploration
- Add explicit reward for using fail-risk signal effectively

### Option 3: Change Architecture
- Make reflex controller less dominant
- Allow strategy layer to have more direct control
- Or simplify to single-layer learned controller

### Option 4: Training Improvements
- Increase entropy coefficient for more exploration
- Use curriculum learning (start easy, increase difficulty)
- Add reward shaping for intermediate goals

## Recommended Next Steps

1. **Reduce reflex controller aggressiveness** (quick fix)
2. **Increase exploration** (higher entropy coefficient)
3. **Tune reward function** (better alignment with EDON score)
4. **Consider architecture simplification** (if above don't work)

