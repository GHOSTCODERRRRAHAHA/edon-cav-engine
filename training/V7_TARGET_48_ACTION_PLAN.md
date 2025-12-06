# EDON v7: Action Plan to Reach Score 48

## Current State Analysis

**Current Score:** ~40.32  
**Target Score:** 48  
**Gap:** +7.68 points

### Current Metrics Breakdown:
- **Interventions:** 41.20/episode → intervention_score = 17.6 (40% weight)
- **Stability:** 0.0199 → stability_score = 80.1 (40% weight)  
- **Length:** 304.8 steps → length_bonus = 6.1 (20% weight)

**Score Formula:**
```
score = 0.4 × intervention_score + 0.4 × stability_score + 0.2 × length_bonus
```

---

## Path to 48: Three Options

### Option 1: Reduce Interventions (MOST IMPACTFUL)
- **Target:** Reduce from 41.2 to ~31.6 interventions (-9.6)
- **Impact:** +7.68 points (0.8 points per intervention reduction)
- **Difficulty:** Medium-Hard (requires better stability control)

### Option 2: Improve Stability
- **Target:** Reduce from 0.0199 to ~0.0009 (-0.019)
- **Impact:** +7.68 points (0.4 points per 0.001 reduction)
- **Difficulty:** Hard (requires very precise control)

### Option 3: Increase Episode Length
- **Target:** Increase from 304.8 to ~1264.8 steps (+960 steps)
- **Impact:** +7.68 points (0.4 points per 50 steps)
- **Difficulty:** Very Hard (episodes typically end due to interventions)

---

## Recommended Strategy: Multi-Pronged Approach

**Best approach:** Combine Option 1 (reduce interventions) with Option 2 (improve stability)

**Target Metrics:**
- Interventions: 41.2 → **35.0** (-6.2) → +4.96 points
- Stability: 0.0199 → **0.0149** (-0.005) → +2.0 points
- Length: 304.8 → **350.0** (+45.2) → +0.36 points
- **Total:** +7.32 points → **Score: ~47.6** (close to 48)

---

## Action Items

### 1. **Tune Reward Function** (CRITICAL)

**File:** `training/edon_score.py`

**Changes to `step_reward()`:**

```python
def step_reward(
    prev_state: Optional[Dict[str, Any]],
    next_state: Dict[str, Any],
    info: Optional[Dict[str, Any]] = None
) -> float:
    # Extract stability metrics
    roll = abs(next_state.get("roll", 0.0))
    pitch = abs(next_state.get("pitch", 0.0))
    roll_velocity = abs(next_state.get("roll_velocity", 0.0))
    pitch_velocity = abs(next_state.get("pitch_velocity", 0.0))
    
    # INCREASE base reward for staying alive (encourage longer episodes)
    reward = 0.2  # Increased from 0.1
    
    # STRONGER penalty for tilt (encourage stability)
    tilt_penalty = (roll + pitch) * 8.0  # Increased from 5.0
    reward -= tilt_penalty
    
    # STRONGER penalty for high velocities (encourage smooth control)
    velocity_penalty = (roll_velocity + pitch_velocity) * 3.0  # Increased from 2.0
    reward -= velocity_penalty
    
    # MUCH STRONGER penalty for interventions (critical!)
    if info:
        if info.get("intervention", False) or info.get("fallen", False):
            reward -= 20.0  # Increased from 10.0
        if info.get("success", False):
            reward += 10.0  # Increased from 5.0
    
    # Reward for low tilt (positive reinforcement)
    if (roll + pitch) < 0.1:  # Very stable
        reward += 0.5
    
    # Reward for low velocity (smooth motion)
    if (roll_velocity + pitch_velocity) < 0.1:
        reward += 0.3
    
    # Clamp to reasonable range
    reward = max(-50.0, min(10.0, reward))
    
    return float(reward)
```

**Rationale:**
- Stronger intervention penalty → model learns to avoid falls
- Stronger tilt/velocity penalties → model learns stability
- Positive rewards for stable states → encourages good behavior
- Higher base reward → encourages longer episodes

---

### 2. **Train Longer and with Better Hyperparameters**

**File:** `training/train_edon_v7.py`

**Recommended Training Command:**

```bash
python training/train_edon_v7.py \
  --episodes 200 \
  --profile high_stress \
  --seed 42 \
  --lr 1e-4 \
  --gamma 0.995 \
  --update-epochs 6 \
  --output-dir models \
  --model-name edon_v7_target48
```

**Hyperparameter Changes:**
- **Episodes:** 20 → 200 (10x more training)
- **Learning Rate:** 3e-4 → 1e-4 (more stable learning)
- **Gamma:** 0.99 → 0.995 (longer horizon, values future rewards more)
- **Update Epochs:** 4 → 6 (more policy refinement per batch)

---

### 3. **Add Reward Shaping for Stability**

**Enhancement to `step_reward()`:**

Add progressive rewards based on stability zones:

```python
# Stability zone rewards
tilt_mag = roll + pitch
if tilt_mag < 0.05:  # Very stable
    reward += 1.0
elif tilt_mag < 0.1:  # Stable
    reward += 0.5
elif tilt_mag < 0.2:  # Warning zone
    reward += 0.1
# Prefall zone gets no bonus (already penalized by tilt_penalty)
```

---

### 4. **Curriculum Learning (Optional but Recommended)**

Train in stages:
1. **Stage 1:** `light_stress` profile (50 episodes) - learn basics
2. **Stage 2:** `medium_stress` profile (50 episodes) - adapt to harder conditions
3. **Stage 3:** `high_stress` profile (100 episodes) - final training

This helps the model learn progressively rather than struggling with high stress from the start.

---

### 5. **Monitor and Iterate**

**During Training:**
- Watch episode scores trend upward
- If interventions aren't decreasing, increase intervention penalty
- If stability isn't improving, increase tilt/velocity penalties
- If episodes are too short, increase base reward

**After Training:**
- Evaluate on 30+ episodes (not just 10)
- Check if score consistently hits 48
- If close but not there, fine-tune reward weights

---

## Implementation Priority

1. **HIGH:** Tune reward function (stronger penalties, positive rewards)
2. **HIGH:** Train longer (200 episodes vs 20)
3. **MEDIUM:** Adjust hyperparameters (lower LR, higher gamma)
4. **MEDIUM:** Add stability zone rewards
5. **LOW:** Curriculum learning (if time permits)

---

## Expected Timeline

- **Reward tuning:** 30 minutes (edit + test)
- **Training:** 2-4 hours (200 episodes)
- **Evaluation:** 30 minutes (multi-seed test)
- **Iteration:** 1-2 more cycles if needed

**Total:** ~4-6 hours to reach 48

---

## Success Criteria

✅ **Score ≥ 48** on 30-episode evaluation  
✅ **Interventions < 35** per episode  
✅ **Stability < 0.015**  
✅ **Consistent across seeds 0-4**

---

## Quick Start

1. Edit `training/edon_score.py` → update `step_reward()`
2. Run: `python training/train_edon_v7.py --episodes 200 --profile high_stress --seed 42 --lr 1e-4 --gamma 0.995 --output-dir models --model-name edon_v7_target48`
3. Evaluate: `python run_eval.py --mode edon --profile high_stress --episodes 30 --seed 42 --output results/edon_v7_target48.json --edon-gain 1.0 --edon-arch v7_learned --edon-score`
4. Check if score ≥ 48, iterate if needed

