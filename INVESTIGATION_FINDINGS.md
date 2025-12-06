# EDON Underperformance Investigation - Key Findings

## Current Performance
- **Baseline**: 39.6 interventions/ep, stability=0.0202, 309.7 steps/ep
- **EDON (gain=0.75)**: 40.0 interventions/ep, stability=0.0206, 306.4 steps/ep
- **Result**: -1.0% interventions, -1.6% stability (WORSE than baseline)

## Root Causes Identified

### 1. **Overly Aggressive Baseline Damping**
**Location**: `evaluation/edon_controller_v3.py`, line 465-510

The mode-based scaling reduces baseline action too aggressively:
- **RECOVERY mode**: `base_scale = 0.55` (45% reduction!)
- **ESCALATE mode**: `base_scale = 0.40` (60% reduction!)
- **BRACE mode**: `base_scale = 0.92` (8% reduction)

**Problem**: When `base_scale < 1.0`, the regulated action becomes:
```
regulated_action = base_scale * baseline_action + corrections
```

Then final action blends:
```
final_action = (1 - edon_gain) * baseline + edon_gain * regulated_action
```

With `edon_gain=0.75` and `base_scale=0.55`:
- Final baseline component = 0.25 * baseline + 0.75 * 0.55 * baseline = **0.6625 * baseline**
- This is a **33.75% reduction** in baseline action, which is too aggressive!

### 2. **Gait Smoothing May Be Destabilizing**
**Location**: `evaluation/edon_controller_v3.py`, line 559

```python
gait_smooth = -GAIT_SMOOTH_GAIN * baseline_action
```

This creates a correction that **opposes** the baseline action. While intended to smooth gait, it may be interfering with stable locomotion.

### 3. **Correction Magnitude May Be Too Small**
**Location**: `evaluation/edon_controller_v3.py`, line 730

The `prefall_ratio` starts at `PREFALL_BASE = 0.22` (22%), but:
- In SAFE zone: `prefall_ratio` is very small (near 0)
- Corrections are only applied in PREFALL/FAIL zones
- If most time is spent in SAFE zone, EDON has minimal effect

### 4. **Episodes Are Shorter**
EDON episodes are **3.3 steps shorter** (306.4 vs 309.7), indicating:
- More early failures
- EDON is causing instability, not preventing it

## Recommended Fixes

### Fix 1: Reduce Baseline Damping
Change mode scaling to be less aggressive:
```python
# RECOVERY: base_scale = 0.75 (was 0.55) - only 25% reduction
# ESCALATE: base_scale = 0.65 (was 0.40) - only 35% reduction
# BRACE: base_scale = 0.95 (was 0.92) - minimal reduction
```

### Fix 2: Increase Correction Strength in PREFALL
- Increase `PREFALL_BASE` from 0.22 to 0.25-0.30
- Ensure corrections are applied even in SAFE zone when risk is elevated

### Fix 3: Remove or Reduce Gait Smoothing
- Either remove `gait_smooth` entirely, or
- Reduce `GAIT_SMOOTH_GAIN` from 0.05 to 0.01

### Fix 4: Verify Correction Direction
- Ensure corrections always oppose tilt (not baseline)
- Add more aggressive direction checking

## Next Steps
1. Implement Fix 1 (reduce baseline damping) - highest priority
2. Test with reduced damping
3. If still underperforming, implement Fix 2 (increase correction strength)
4. Re-run comprehensive tests

