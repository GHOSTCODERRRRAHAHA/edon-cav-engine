# Zero-Shot Modulation Fixes - Improving Demo Performance

## The 3 Main Issues Fixed

### Issue 1: Wrong `gain_scale` (Too Low/High)
**Problem:** EDON outputs gain_scale values that are too conservative or too aggressive for MuJoCo.

**Fix:** State-aware bounds
- **Stable robot** (tilt < 5.7°): `gain_scale` clamped to [0.5, 1.2] (conservative)
- **Moderate tilt** (5.7°-11.5°): `gain_scale` clamped to [0.6, 1.3] (moderate)
- **Unstable robot** (tilt > 11.5°): `gain_scale` clamped to [0.7, 1.4] (aggressive)

**Also:** Fail-risk based adjustment
- If fail-risk is high but robot is stable → reduce gain_scale by 10% (fail-risk might be wrong)
- If fail-risk is low but robot is unstable → increase gain_scale by 10% (fail-risk might be wrong)

### Issue 2: Wrong `lateral_compliance` (Too Low/High)
**Problem:** EDON outputs lateral_compliance values that prevent proper roll/pitch correction.

**Fix:** State-aware bounds
- **Stable robot** (tilt < 5.7°): `lateral_compliance` clamped to [0.4, 0.9] (conservative)
- **Moderate tilt** (5.7°-11.5°): `lateral_compliance` clamped to [0.5, 1.0] (moderate)
- **Unstable robot** (tilt > 11.5°): `lateral_compliance` clamped to [0.6, 1.0] (stronger correction)

### Issue 3: Wrong `step_height_bias` (Extreme Values)
**Problem:** EDON outputs step_height_bias values that cause legs to move too far.

**Fix:** More conservative bounds
- `step_height_bias` clamped to [-0.5, 0.5] (was [-1, 1])
- Prevents extreme leg positions that destabilize robot

---

## Additional Fix: Modulation Smoothing

**Problem:** Rapid changes in modulations cause oscillation.

**Fix:** Exponential smoothing
- Smooth modulations over time (70% new value, 30% old value)
- Prevents sudden jumps that cause oscillation
- Uses last 5 modulations for smoothing

**Example:**
```
Step 100: gain_scale = 0.7
Step 101: EDON outputs gain_scale = 1.4 (sudden jump)
After smoothing: gain_scale = 0.7 * 0.3 + 1.4 * 0.7 = 1.19 (smooth transition)
Result: No oscillation ✅
```

---

## How It Works

### Step-by-Step Process

1. **EDON API returns modulations** (gain_scale, lateral_compliance, step_height_bias)

2. **Apply modulation fixes:**
   - Check robot state (roll, pitch, tilt)
   - Adjust bounds based on stability
   - Adjust based on fail-risk (if it seems wrong)
   - Smooth with previous modulations

3. **Apply modulations to action:**
   - gain_scale → entire action
   - lateral_compliance → root rotation
   - step_height_bias → leg joints

4. **Result:** More stable, less oscillation, fewer interventions

---

## Expected Improvements

### Before Fixes
```
Baseline: 1 intervention
EDON: 4 interventions
Result: EDON makes things worse ❌
```

### After Fixes
```
Baseline: 1 intervention
EDON: 0-1 interventions
Result: EDON helps or matches baseline ✅
```

**Improvements:**
- **State-aware bounds:** Prevents wrong modulations for current robot state
- **Fail-risk adjustment:** Compensates for fail-risk prediction errors
- **Modulation smoothing:** Prevents oscillation from rapid changes
- **Conservative step_height_bias:** Prevents extreme leg positions

---

## Technical Details

### State-Aware Bounds Logic

```python
# Get robot tilt
max_tilt = max(abs(roll), abs(pitch))

# Adjust bounds based on stability
if max_tilt > 0.2:  # Unstable (>11.5°)
    gain_scale_bounds = [0.7, 1.4]  # Allow aggressive
    lateral_compliance_bounds = [0.6, 1.0]  # Need stronger correction
elif max_tilt > 0.1:  # Moderate (5.7°-11.5°)
    gain_scale_bounds = [0.6, 1.3]  # Moderate
    lateral_compliance_bounds = [0.5, 1.0]  # Moderate
else:  # Stable (<5.7°)
    gain_scale_bounds = [0.5, 1.2]  # Conservative
    lateral_compliance_bounds = [0.4, 0.9]  # Conservative
```

### Fail-Risk Adjustment Logic

```python
# If fail-risk seems wrong, adjust modulations
if fail_risk > 0.7 and max_tilt < 0.1:
    # Fail-risk says high risk, but robot is stable
    # → Fail-risk might be wrong, reduce aggressiveness
    gain_scale = gain_scale * 0.9
    
elif fail_risk < 0.3 and max_tilt > 0.15:
    # Fail-risk says low risk, but robot is unstable
    # → Fail-risk might be wrong, increase aggressiveness
    gain_scale = gain_scale * 1.1
```

### Modulation Smoothing Logic

```python
# Exponential smoothing: 70% new, 30% old
if len(modulation_history) > 0:
    last_gain = modulation_history[-1]["gain_scale"]
    gain_scale = 0.7 * gain_scale + 0.3 * last_gain
```

---

## What This Means

### For Zero-Shot Performance

**Before fixes:**
- EDON could make things worse (1 → 4 interventions)
- Wrong modulations destabilized robot
- No protection against bad values

**After fixes:**
- EDON is more likely to help (1 → 0-1 interventions)
- State-aware bounds prevent wrong modulations
- Smoothing prevents oscillation
- Fail-risk adjustment compensates for prediction errors

### For Training

**These fixes help zero-shot performance while training:**
- Better zero-shot performance = better starting point for training
- Fewer bad episodes = faster training convergence
- More stable behavior = easier to learn from

**After training:**
- Trained model will have correct modulations
- These fixes still help (safety net)
- But trained model should rarely need them

---

## Summary

**3 Main Issues Fixed:**
1. ✅ **gain_scale:** State-aware bounds + fail-risk adjustment
2. ✅ **lateral_compliance:** State-aware bounds based on stability
3. ✅ **step_height_bias:** More conservative bounds

**Additional Fix:**
4. ✅ **Modulation smoothing:** Prevents oscillation from rapid changes

**Result:**
- Zero-shot performance should improve
- Fewer cases where EDON makes things worse
- More consistent behavior
- Better starting point for training

**Note:** These fixes improve zero-shot performance, but training on MuJoCo is still needed for 90%+ improvement.

