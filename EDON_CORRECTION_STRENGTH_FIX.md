# EDON Correction Strength Fix

## Problem

1. **EDON state mapping was broken**: Always returned RESTORATIVE, so state-aware logic wasn't working
2. **Corrections too weak**: Only ~6% of baseline norm (target: 15-20% in PREFALL)
3. **State wasn't varying**: Debug logs showed only RESTORATIVE state

## Solution

### 1. Fixed State Mapping (`evaluation/edon_state.py`)

**Before:**
- Used `state_class` as primary indicator
- If `state_class` was "restorative" or "unknown", always returned RESTORATIVE
- Fallback to p_stress/p_chaos only if state_class missing

**After:**
- **PRIMARY**: Use `p_stress` and `p_chaos` (continuous metrics) to infer state
- **SECONDARY**: Use `state_class` only if p_stress/p_chaos are missing
- Threshold-based mapping:
  - OVERLOAD: p_chaos > 0.65
  - STRESS: p_stress > 0.55
  - RESTORATIVE: p_stress < 0.25 AND p_chaos < 0.15
  - FOCUS: p_stress < 0.45 AND p_chaos < 0.25
  - BALANCED: default

This ensures state actually varies based on EDON's continuous metrics.

### 2. Target Ratio Scaling (Replaced Gain-Based Approach)

**Before:**
- Used gain multipliers (Kp_scale, Kd_scale) that scaled base gains
- Corrections were ~6% of baseline norm (too weak)
- No explicit target for correction magnitude

**After:**
- **Target ratio approach**: Explicitly target correction magnitude as % of baseline
- **SAFE zone**: target_ratio = 0.05 (5% of baseline)
- **PREFALL zone**: target_ratio = 0.18 (18% of baseline - much stronger!)
- **FAIL zone**: target_ratio = 0.05 (minimal, environment intervenes)

After computing PD corrections, rescale to hit target:
```python
if baseline_norm > 1e-6 and corr_norm > 1e-6:
    desired_corr_norm = target_ratio * baseline_norm
    scale = desired_corr_norm / corr_norm
    scale = np.clip(scale, 0.1, 5.0)  # Avoid insane scaling
    correction *= scale
```

### 3. State as Modulator (Not Multiplier)

**Before:**
- `get_state_gains()` returned (Kp_scale, Kd_scale, speed_scale)
- State multiplied gains directly (could be 2.0-3.0x)

**After:**
- `get_state_modulation()` returns (ratio_scale, speed_scale)
- State modulates `target_ratio`, not gains
- Max ratio_scale = 1.3x (conservative)

**State modulation in PREFALL:**
- BALANCED: ratio_scale = 1.0, speed_scale = 0.9
- STRESS: ratio_scale = 1.15, speed_scale = 0.85
- OVERLOAD: ratio_scale = 1.3, speed_scale = 0.8
- EMERGENCY: ratio_scale = 1.3, speed_scale = 0.7
- FOCUS: ratio_scale = 1.05, speed_scale = 0.95
- RESTORATIVE: ratio_scale = 1.05, speed_scale = 0.9

**State modulation in SAFE:**
- STRESS/OVERLOAD: ratio_scale = 1.05 (tiny tweak)
- Others: ratio_scale = 1.0 (no change)

### 4. Example Correction Magnitudes

**Before (broken):**
- Baseline norm: ~0.33
- Correction norm: ~0.02 (6% of baseline)
- Result: EDON barely visible

**After (fixed):**
- Baseline norm: ~0.33
- **SAFE zone**: Correction norm: ~0.016 (5% of baseline)
- **PREFALL zone**: Correction norm: ~0.06 (18% of baseline)
- **PREFALL + STRESS**: Correction norm: ~0.069 (18% * 1.15 = 20.7% of baseline)
- **PREFALL + OVERLOAD**: Correction norm: ~0.077 (18% * 1.3 = 23.4% of baseline)

## Expected Results

1. **State diversity**: Debug logs should show mix of BALANCED, FOCUS, STRESS, OVERLOAD, RESTORATIVE
2. **Stronger corrections in PREFALL**: ~18-23% of baseline (vs 6% before)
3. **Minimal corrections in SAFE**: ~5% of baseline (unchanged)
4. **Better intervention reduction**: EDON should now have visible impact

## Testing

```bash
# Test with debug logging to verify state diversity
python run_eval.py --mode edon --episodes 30 --profile medium_stress --edon-gain 0.75 --debug-edon --output results/edon_fixed.json

# Compare with baseline
python plot_results.py --baseline results/baseline_medium.json --edon results/edon_fixed.json --output plots/fixed
```

**Look for in debug logs:**
- State histogram showing multiple states (not just RESTORATIVE)
- Correction norms: ~0.06 in PREFALL (vs ~0.02 before)
- Intervention reduction: Should see 10-20% improvement

## Files Modified

1. **`evaluation/edon_state.py`**: Fixed state mapping to use p_stress/p_chaos as primary indicators
2. **`run_eval.py`**: 
   - Replaced `get_state_gains()` with `get_state_modulation()`
   - Added target ratio scaling (5% SAFE, 18% PREFALL)
   - State modulates target_ratio, not gains

## Key Design Principles

1. **State mapping uses continuous metrics**: p_stress and p_chaos are more reliable than state_class
2. **Target ratio approach**: Explicit control over correction magnitude
3. **State as modulator**: Max 1.3x boost, not 2.0-3.0x multipliers
4. **Zone-aware strength**: Strong in PREFALL (18%), weak in SAFE (5%)

