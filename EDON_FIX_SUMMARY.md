# EDON Controller Fix - Conservative State-Aware Control

## Problem

The state-aware EDON controller was making things worse:
- Baseline: 39.07 interventions/episode, stability 0.0196
- EDON: 41.17 interventions/episode (+5.4% worse), stability 0.0213 (~9% worse)

Root causes:
1. **State scaling too aggressive**: 2.0-3.0x multipliers in STRESS/OVERLOAD
2. **State logic active in SAFE zone**: Injecting noise into stable gait
3. **Uncalibrated gains**: Old gains now scaled by state multipliers → effective gains too high

## Solution

### 1. Conservative State Scaling

**Before:**
- OVERLOAD in prefall: Kp_scale = 2.5x, Kd_scale = 2.0x
- STRESS in prefall: Kp_scale = 2.0x, Kd_scale = 1.8x
- EMERGENCY: Kp_scale = 3.0x, Kd_scale = 2.5x

**After:**
- OVERLOAD in prefall: Kp_scale = 1.3x (max 30% boost)
- STRESS in prefall: Kp_scale = 1.2x (max 20% boost)
- EMERGENCY: Kp_scale = 1.3x (same as overload)
- **Safe zone**: State barely matters (Kp_scale = 1.0-1.05x)

### 2. Safe Zone Minimalism

**Before:**
- Safe zone had variable scaling (0.5-1.1x) based on state

**After:**
- Safe zone: Kp_scale = 1.0-1.05x (minimal intervention)
- Don't let EDON inject noise into otherwise stable gait
- State only matters in PREFALL zone

### 3. Safety Clamping

**Before:**
- max_ratio = 1.3 (30% extra magnitude)

**After:**
- max_ratio = 1.2 (20% extra magnitude) - more conservative
- Prevents EDON from overpowering baseline

### 4. Simple PD Mode (Testing)

Added `--simple-pd` flag to test PD controller without state scaling:
```bash
python run_eval.py --mode edon --episodes 30 --profile medium_stress --edon-gain 0.75 --simple-pd --output results/edon_simple_pd.json
```

This helps verify that PD itself works before reintroducing state logic.

### 5. Debug Logging

Added `--debug-edon` flag for per-episode stats:
- % of steps in SAFE vs PREFALL vs FAIL
- Mean baseline action norm
- Mean correction norm
- Histogram of EDON states (STRESS/OVERLOAD frequency)

```bash
python run_eval.py --mode edon --episodes 30 --profile medium_stress --edon-gain 0.75 --debug-edon --output results/edon_debug.json
```

## Testing Strategy

### Step 1: Verify Simple PD Works
```bash
# Simple PD (no state scaling)
python run_eval.py --mode edon --episodes 30 --profile medium_stress --edon-gain 0.75 --simple-pd --output results/edon_simple_pd.json

# Compare with baseline
python plot_results.py --baseline results/baseline_medium.json --edon results/edon_simple_pd.json --output plots/simple_pd
```

**Expected**: Should be ~+1-5% better than baseline (like before state-aware changes)

### Step 2: Reintroduce State (Conservative)
```bash
# State-aware (conservative scaling)
python run_eval.py --mode edon --episodes 30 --profile medium_stress --edon-gain 0.75 --output results/edon_state_aware.json

# Compare with baseline
python plot_results.py --baseline results/baseline_medium.json --edon results/edon_state_aware.json --output plots/state_aware
```

**Expected**: Should be better than simple PD, but not worse than baseline

### Step 3: Debug Analysis
```bash
# With debug logging
python run_eval.py --mode edon --episodes 30 --profile medium_stress --edon-gain 0.75 --debug-edon --output results/edon_debug.json
```

**Look for:**
- "EDON spent a ton of time in STRESS/OVERLOAD while in SAFE zone + big correction norms" → overreacting too early
- "Correction norm is often > baseline norm" → EDON overpowering baseline

## Key Changes in Code

1. **`get_state_gains()`**: Conservative scaling (1.0-1.3x max, only in PREFALL)
2. **`apply_edon_regulation()`**: Added `use_state` parameter for simple PD mode
3. **Safety clamp**: Reduced from 1.3x to 1.2x
4. **Debug logging**: Per-episode stats for zone distribution, action norms, state counts

## Files Modified

- `run_eval.py`: Conservative state scaling, simple PD mode, debug logging
- `evaluation/edon_state.py`: (unchanged) State enum and mapping

## Next Steps

1. Run simple PD test to verify PD works
2. Run state-aware test with conservative scaling
3. Analyze debug logs to identify any remaining issues
4. Tune base gains (`EDON_BASE_KP_*`) if needed

