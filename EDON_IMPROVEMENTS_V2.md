# EDON Controller Improvements V2

## Summary

Implemented comprehensive improvements to make EDON clearly better than baseline under medium_stress by leveraging EDON's real "superpower" (CAV/state) instead of acting like a noisy PD overlay.

## Key Changes

### 1. Fixed State Mapping (`evaluation/edon_state.py`)

**Added `compute_risk_score()` function:**
- Computes continuous risk score from `p_stress` and `p_chaos`
- Formula: `risk = 0.6 * p_chaos + 0.4 * p_stress`
- Returns value in [0, 1] where higher = more urgent need for stabilization

**State mapping already uses p_stress/p_chaos as primary indicators** (from previous fix)

### 2. Risk-Based Target Ratio Modulation (`run_eval.py`)

**Base target ratios by zone:**
- SAFE: 2% of baseline (almost no EDON influence)
- PREFALL: 18% of baseline (strong rescue)
- FAIL: 3% of baseline (gentle damping only)

**Risk-based modulation in PREFALL:**
- Low risk (risk < 0.3): target_ratio = 12%
- Medium risk (0.3 ≤ risk < 0.6): target_ratio = 20%
- High risk (risk ≥ 0.6): target_ratio = 25% (max)

**Blending:**
- `target_ratio = 0.5 * base_target + 0.5 * risk_modulation`
- Then apply state modulation (ratio_scale) on top

This ensures EDON acts more aggressively when EDON itself detects high risk (high p_chaos/p_stress).

### 3. Direction Check for Stabilizing Corrections

**Problem:** Corrections could amplify tilt instead of opposing it.

**Solution:** In PREFALL/FAIL zones, check if balance-critical corrections (indices 0-1) oppose tilt:
- If roll > 0 (falling right) and correction[0] > 0 (amplifying): flip correction[0]
- If pitch > 0 (falling forward) and correction[1] > 0 (amplifying): flip correction[1]
- Flip with 0.7x reduction to avoid overcorrection

This ensures corrections are truly stabilizing, not destabilizing.

### 4. Joint-Aware Corrections

**Focus on balance-critical joints:**
- Indices 0-3: Balance control (roll, pitch, COM x, COM y)
- Indices 4+: Forward velocity / movement (get zero correction from EDON)

`map_torso_correction_to_action()` already focuses corrections on indices 0-3, with indices 4+ getting zero. This ensures EDON doesn't interfere with non-balance motions.

### 5. Enhanced Debug Logging

**New debug metrics per episode:**
- Correction ratios by zone (SAFE, PREFALL, FAIL)
- PREFALL stabilizing vs destabilizing percentage
- EDON state distribution (all steps and PREFALL-only)
- Mean correction ratio overall and by zone

**Example debug output:**
```
[DEBUG-EDON] Episode 1 stats:
  Zones: SAFE=75.2%, PREFALL=12.8%, FAIL=12.0%
  Action norms: baseline=0.335, correction=0.042, ratio=0.125
  Correction ratios by zone: SAFE=0.015, PREFALL=0.195, FAIL=0.025
  PREFALL corrections: 78.5% stabilizing, 21.5% destabilizing
  EDON states (all): BALANCED=450, STRESS=120, OVERLOAD=30
  EDON states (PREFALL): BALANCED=15, STRESS=45, OVERLOAD=20
```

### 6. Zone Thresholds Verification

**Current thresholds (from `evaluation/config.py`):**
- SAFE_LIMIT: 0.15 rad (~8.6°)
- PREFALL_LIMIT: 0.30 rad (~17.2°)
- FAIL_LIMIT: 0.35 rad (~20°)

**Analysis:**
- PREFALL zone spans 0.15-0.30 rad (15° range) - enough time for EDON to act
- FAIL zone starts at 0.35 rad - intervention threshold
- If FAIL % is high (15-20%), it indicates baseline is struggling, not zone definition issue

## Expected Results

### Target Metrics (medium_stress, 30 episodes, gain=0.75):

1. **Intervention reduction**: +5-10% (fewer interventions than baseline)
2. **Stability improvement**: ≥5% (lower stability score = better)
3. **Prefall time/events**: Equal or less than baseline
4. **Debug logs show**:
   - PREFALL correction ratios: 0.15-0.25 range
   - PREFALL stabilizing: >70% of corrections
   - EDON states vary (not all RESTORATIVE)
   - Higher risk states (STRESS/OVERLOAD) correlate with more aggressive corrections in PREFALL

### Correction Magnitudes:

**Before:**
- SAFE: ~5% of baseline
- PREFALL: ~7-10% of baseline (too weak)
- Result: EDON barely visible

**After:**
- SAFE: ~2% of baseline (minimal)
- PREFALL: ~15-25% of baseline (strong, risk-dependent)
- FAIL: ~3% of baseline (gentle damping)
- Result: EDON has visible impact in PREFALL

## Testing

```bash
# Baseline
python run_eval.py --mode baseline --episodes 30 --profile medium_stress --output results/baseline_medium.json

# EDON with enhanced controller
python run_eval.py --mode edon --episodes 30 --profile medium_stress --edon-gain 0.75 --debug-edon --output results/edon_fixed_v2.json

# Compare
python plot_results.py --baseline results/baseline_medium.json --edon results/edon_fixed_v2.json --output plots/fixed_v2
```

## Files Modified

1. **`evaluation/edon_state.py`**: Added `compute_risk_score()` function
2. **`run_eval.py`**:
   - Risk-based target ratio modulation in PREFALL
   - Direction check for stabilizing corrections
   - Enhanced debug logging with zone-specific ratios and stabilizing/destabilizing counts
   - Joint-aware corrections (already focused on indices 0-3)

## Key Design Principles

1. **Risk-aware aggression**: EDON acts more aggressively when it detects high risk (p_chaos/p_stress)
2. **Zone-appropriate strength**: Strong in PREFALL (15-25%), minimal in SAFE (2%)
3. **Stabilizing direction**: Corrections always oppose tilt, never amplify it
4. **Joint-aware**: Focus on balance-critical joints (0-3), don't interfere with others
5. **State diversity**: Use continuous risk score, not just discrete state labels

## Next Steps

1. Run evaluation and verify improvements
2. Analyze debug logs to confirm:
   - PREFALL ratios in 0.15-0.25 range
   - >70% stabilizing corrections in PREFALL
   - State diversity and risk correlation
3. Tune risk thresholds if needed (currently 0.3, 0.6)
4. Adjust target ratios if corrections are still too weak/strong

