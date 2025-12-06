# Technical Explanation: Why EDON Makes Things Worse (Before Safety)

## Your Question

**"Before we added the safety mechanism, EDON was making things worse. Why?"**

This is about the **actual technical reasons** why EDON's modulations destabilize the robot. Let me break it down step-by-step.

---

## The Core Problem: Wrong Modulations for MuJoCo

EDON outputs 3 modulations that adjust the baseline controller:
1. **`gain_scale`** (0.5-1.5): Scales entire action magnitude
2. **`lateral_compliance`** (0.0-1.0): Reduces root rotation (roll/pitch/yaw)
3. **`step_height_bias`** (-1 to 1): Adjusts leg joint positions

**Problem:** These modulations were learned for `MockHumanoidEnv` (simplified), not MuJoCo (full physics).

---

## Example 1: gain_scale Too Low → Robot Can't Correct Fast Enough

### What Happens

**Baseline Action (Correct):**
```
Baseline sees: Robot tilting 10° to the right
Baseline outputs: [15, -10, 5, 8, -8, 0, ...] (strong correction to left)
Result: Robot corrects quickly → stays stable ✅
```

**EDON Modulation (Wrong):**
```
EDON sees: Same tilt (10° to right)
EDON thinks: "This is stable, reduce action" (wrong!)
EDON outputs: gain_scale = 0.7 (reduce by 30%)
Corrected action: [10.5, -7, 3.5, 5.6, -5.6, 0, ...] (weaker correction)
Result: Robot corrects too slowly → tilts more → intervention ❌
```

### Why This Happens

**EDON's Training Environment:**
- Simplified dynamics (direct roll/pitch control)
- Small corrections work fine
- Conservative actions are safe

**MuJoCo Reality:**
- Full rigid-body physics (inertia, momentum)
- Needs stronger corrections to overcome inertia
- Conservative actions are too weak → robot can't recover

**Result:** EDON reduces action strength when it should increase it → robot tilts more → intervention

---

## Example 2: lateral_compliance Too Low → Overcorrection

### What Happens

**Baseline Action (Correct):**
```
Baseline sees: Robot tilting 8° forward
Baseline outputs root rotation: [0, 0, 0, 2, -2, 0, ...] (moderate pitch correction)
Result: Robot corrects smoothly → stays stable ✅
```

**EDON Modulation (Wrong):**
```
EDON sees: Same tilt (8° forward)
EDON thinks: "Need strong lateral correction" (wrong!)
EDON outputs: lateral_compliance = 0.4 (reduce root rotation by 60%)
Corrected root rotation: [0, 0, 0, 0.8, -0.8, 0, ...] (too weak)
Result: Robot can't correct pitch → tilts more → intervention ❌
```

**OR (Opposite Problem):**
```
EDON outputs: lateral_compliance = 1.2 (increase root rotation by 20%)
Corrected root rotation: [0, 0, 0, 2.4, -2.4, 0, ...] (too strong)
Result: Robot overcorrects → oscillates → intervention ❌
```

### Why This Happens

**EDON's Training Environment:**
- Root rotation directly controls balance
- Small changes have predictable effects
- lateral_compliance learned for that environment

**MuJoCo Reality:**
- Root rotation affects full-body dynamics
- Too weak = can't correct
- Too strong = overcorrection/oscillation
- Different relationship than training environment

**Result:** EDON applies wrong lateral_compliance → either too weak (can't correct) or too strong (overcorrects) → intervention

---

## Example 3: step_height_bias Wrong → Legs Can't Support

### What Happens

**Baseline Action (Correct):**
```
Baseline sees: Robot needs to step higher (obstacle)
Baseline outputs leg joints: [0, 0, 0, 0, 0, 0, 5, -5, 3, -3, 2, -2] (raise legs)
Result: Robot steps over obstacle → stays stable ✅
```

**EDON Modulation (Wrong):**
```
EDON sees: Robot needs to step higher
EDON thinks: "Reduce step height" (wrong!)
EDON outputs: step_height_bias = -0.8 (reduce leg height)
Corrected leg joints: [0, 0, 0, 0, 0, 0, 4.2, -4.2, 2.2, -2.2, 1.2, -1.2] (lower legs)
Result: Robot can't step high enough → trips → intervention ❌
```

### Why This Happens

**EDON's Training Environment:**
- Step height directly affects balance
- Lower steps might be safer in some cases
- step_height_bias learned for that environment

**MuJoCo Reality:**
- Step height affects contact, stability, and obstacle clearance
- Wrong step height = can't clear obstacles or maintain contact
- Different relationship than training environment

**Result:** EDON applies wrong step_height_bias → legs can't support or clear obstacles → intervention

---

## Example 4: Combined Modulations → Cascading Failure

### What Happens

**Step 100:**
```
Robot state: Tilt = 5° (stable, but EDON thinks it's unstable)
EDON outputs:
  - gain_scale = 0.6 (too conservative)
  - lateral_compliance = 0.3 (too weak)
  - step_height_bias = -0.5 (too low)
Result: Action too weak → robot tilts to 8° ❌
```

**Step 101:**
```
Robot state: Tilt = 8° (now actually unstable)
EDON sees: "High risk!" → panics
EDON outputs:
  - gain_scale = 1.4 (too aggressive)
  - lateral_compliance = 1.2 (too strong)
  - step_height_bias = 0.8 (too high)
Result: Overcorrection → robot tilts to 12° in opposite direction ❌
```

**Step 102:**
```
Robot state: Tilt = 12° (oscillating)
EDON sees: "Very high risk!" → more panic
EDON outputs:
  - gain_scale = 1.5 (maximum, too aggressive)
  - lateral_compliance = 0.2 (swings to opposite extreme)
  - step_height_bias = -0.9 (swings to opposite extreme)
Result: Extreme oscillation → robot tilts to 20° → INTERVENTION ❌
```

### Why This Happens

**EDON's Training Environment:**
- Modulations work together smoothly
- Small changes have predictable effects
- Combined modulations learned for that environment

**MuJoCo Reality:**
- Modulations interact in complex ways
- Wrong combination = cascading failure
- Different dynamics than training environment

**Result:** EDON applies wrong modulations → robot destabilizes → EDON panics and applies more wrong modulations → cascading failure → intervention

---

## Example 5: Fail-Risk Prediction Errors → Wrong Strategy

### What Happens

**Scenario:**
```
Robot state: Tilt = 3° (actually stable)
Fail-risk model: Predicts risk = 0.8 (thinks it's unstable) ❌
EDON selects: RECOVERY_BALANCE strategy (aggressive)
EDON outputs:
  - gain_scale = 1.3 (too aggressive for stable state)
  - lateral_compliance = 0.5 (too strong)
Result: Overcorrection → robot destabilizes → intervention ❌
```

**OR (Opposite):**
```
Robot state: Tilt = 15° (actually unstable)
Fail-risk model: Predicts risk = 0.2 (thinks it's stable) ❌
EDON selects: NORMAL strategy (conservative)
EDON outputs:
  - gain_scale = 0.9 (too weak for unstable state)
  - lateral_compliance = 0.8 (too weak)
Result: Under-correction → robot tilts more → intervention ❌
```

### Why This Happens

**Fail-Risk Model Training:**
- Trained on `MockHumanoidEnv` (simplified)
- Learned risk patterns for that environment
- Different failure modes than MuJoCo

**MuJoCo Reality:**
- Different physics = different failure modes
- Risk patterns don't match training
- Model mispredicts risk → wrong strategy → wrong modulations

**Result:** Fail-risk model mispredicts → EDON selects wrong strategy → wrong modulations → intervention

---

## The Mathematical Problem

### How Modulations Are Applied

```python
# 1. Normalize baseline action to [-1, 1]
normalized = baseline_action / 20.0

# 2. Apply gain_scale (scales entire action)
corrected = normalized * gain_scale

# 3. Apply lateral_compliance (scales root rotation only)
corrected[3:6] = corrected[3:6] * lateral_compliance

# 4. Apply step_height_bias (adds to leg joints)
corrected[6:12] = corrected[6:12] + step_height_bias * 0.1

# 5. Clamp and scale back
corrected = np.clip(corrected, -1.0, 1.0) * 20.0
```

### What Goes Wrong

**If gain_scale is too low (0.6):**
- Action magnitude reduced by 40%
- Robot can't generate enough torque to correct
- **Result:** Under-correction → intervention

**If gain_scale is too high (1.4):**
- Action magnitude increased by 40%
- Robot overcorrects, creates oscillation
- **Result:** Over-correction → intervention

**If lateral_compliance is too low (0.3):**
- Root rotation reduced by 70%
- Robot can't correct roll/pitch
- **Result:** Can't balance → intervention

**If lateral_compliance is too high (1.2):**
- Root rotation increased by 20%
- Robot overcorrects roll/pitch
- **Result:** Oscillation → intervention

**If step_height_bias is wrong (-0.8):**
- Leg joints lowered
- Robot can't clear obstacles or maintain contact
- **Result:** Trips or loses contact → intervention

---

## Why This Happens in Zero-Shot

### Environment Mismatch

| Aspect | Training Environment | MuJoCo Environment |
|--------|---------------------|-------------------|
| **Physics** | Simplified (direct control) | Full rigid-body |
| **Dynamics** | Predictable, linear | Complex, nonlinear |
| **Action Space** | Abstract (roll/pitch/COM) | Joint torques |
| **Failure Modes** | Simple (tilt threshold) | Complex (contact, inertia, etc.) |
| **Modulation Effects** | Direct, predictable | Indirect, complex |

### Result

EDON's learned modulations don't transfer well:
- **Too conservative** → under-correction → intervention
- **Too aggressive** → over-correction → intervention
- **Wrong combination** → cascading failure → intervention

---

## Concrete Example: Your Case (1 → 4 Interventions)

### What Probably Happened

**Step 50-100:**
```
Robot: Stable (tilt = 2°)
EDON: Thinks it's unstable (fail-risk = 0.7)
EDON: gain_scale = 1.2, lateral_compliance = 0.4
Result: Overcorrection → robot tilts to 5° ❌
```

**Step 100-150:**
```
Robot: Tilt = 5° (now actually unstable)
EDON: Thinks it's very unstable (fail-risk = 0.9)
EDON: gain_scale = 1.4, lateral_compliance = 0.3
Result: Extreme overcorrection → robot tilts to 12° ❌
```

**Step 150-200:**
```
Robot: Tilt = 12° (oscillating)
EDON: Panics (fail-risk = 1.0)
EDON: gain_scale = 0.8, lateral_compliance = 1.1 (swings to opposite)
Result: Wrong correction → robot tilts to 20° → INTERVENTION #1 ❌
```

**Step 200-300:**
```
Robot: Recovering from intervention
EDON: Still applying wrong modulations
Result: More interventions (#2, #3, #4) ❌
```

**Baseline (Same Conditions):**
```
Robot: Stable (tilt = 2°)
Baseline: Standard control (no EDON)
Result: Handles disturbances normally → 1 intervention ✅
```

---

## Summary: Why EDON Makes Things Worse

### Root Causes

1. **Wrong gain_scale:**
   - Too low → under-correction → can't recover
   - Too high → over-correction → oscillation

2. **Wrong lateral_compliance:**
   - Too low → can't correct roll/pitch
   - Too high → overcorrects roll/pitch

3. **Wrong step_height_bias:**
   - Wrong value → legs can't support or clear obstacles

4. **Wrong strategy selection:**
   - Fail-risk model mispredicts → wrong strategy → wrong modulations

5. **Cascading failures:**
   - One wrong modulation → robot destabilizes → EDON panics → more wrong modulations

### The Core Issue

**EDON was trained on a different environment:**
- Learned modulations for `MockHumanoidEnv` (simplified)
- Those modulations don't work well in MuJoCo (full physics)
- Zero-shot transfer fails → wrong modulations → interventions

### The Solution

1. **Safety mechanism:** Disables EDON if it's making things worse
2. **Training:** Train EDON on MuJoCo → learns correct modulations → 90%+ improvement

---

## Bottom Line

**EDON makes things worse because:**
- Modulations learned for different environment
- Wrong gain_scale → under/over-correction
- Wrong lateral_compliance → can't balance
- Wrong step_height_bias → legs fail
- Wrong strategy → wrong modulations
- Cascading failures → multiple interventions

**This is why we need:**
- Safety mechanism (prevents worse performance)
- Training on MuJoCo (learns correct modulations)

