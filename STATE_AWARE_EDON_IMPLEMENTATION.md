# State-Aware EDON Controller Implementation

## Overview

Refactored EDON controller to use EDON's internal state as a "superpower" to aggressively prevent falls in prefall zones while staying gentle in safe zones.

## Key Changes

### 1. EDON State Enum (`evaluation/edon_state.py`)

Created `EdonState` enum and `map_edon_output_to_state()` function to map raw EDON output to clean state classifications:

- `BALANCED`: Normal operation
- `FOCUS`: Optimal performance, moderate stress
- `STRESS`: High stress, caution required (maps to "alert")
- `OVERLOAD`: Very high stress/chaos, performance degraded
- `RESTORATIVE`: Low stress, recovery mode
- `EMERGENCY`: Extremely high stress/chaos, safety mode

### 2. State-Aware Gain Selection (`get_state_gains()`)

Returns `(Kp_scale, Kd_scale, speed_scale)` based on EDON state and tilt zone:

**Safe Zone:**
- FOCUS: Kp=1.1, speed=1.0 (slight precision boost)
- RESTORATIVE: Kp=0.8, speed=0.95 (gentle)
- Others: Kp=0.5, speed=1.0 (minimal intervention)

**Prefall Zone (EDON's Superpower):**
- OVERLOAD: Kp=2.5, Kd=2.0, speed=0.6 (maximum stabilization)
- STRESS: Kp=2.0, Kd=1.8, speed=0.7 (aggressive)
- EMERGENCY: Kp=3.0, Kd=2.5, speed=0.5 (maximum response)
- FOCUS: Kp=1.5, Kd=1.3, speed=0.95 (precision mode)
- BALANCED: Kp=1.3, Kd=1.2, speed=0.9 (moderate)
- RESTORATIVE: Kp=1.1, Kd=1.0, speed=0.85 (gentle recovery)

**Fail Zone:**
- No special handling (intervention already triggered)

### 3. Hyperparameter Configuration

Added config block at top of `run_eval.py`:

```python
EDON_BASE_KP_ROLL = 0.08
EDON_BASE_KP_PITCH = 0.08
EDON_BASE_KD_ROLL = 0.02
EDON_BASE_KD_PITCH = 0.02
EDON_MAX_CORRECTION_RATIO = 1.3
EDON_PREFALL_SPEED_SCALE_STRESS = 0.7
EDON_PREFALL_SPEED_SCALE_OVERLOAD = 0.6
EDON_PREFALL_SPEED_SCALE_FOCUS = 0.95
EDON_PREFALL_SPEED_SCALE_BALANCED = 0.9
```

### 4. Core Controller Functions

**`map_torso_correction_to_action()`**: Maps roll/pitch corrections to action space (indices 0-3 for balance, 4+ for movement)

**`apply_forward_speed_scale()`**: Applies speed scaling to forward velocity component (indices 4+)

**`clamp_action_relative_to_baseline()`**: Safety clamp to prevent EDON from exceeding baseline by more than `max_ratio` (1.3 in prefall, 1.2 in safe)

**`apply_edon_regulation()`**: Main state-aware regulation function:
1. Maps EDON output to `EdonState` enum
2. Classifies tilt zone (safe/prefall/fail)
3. Gets state-dependent gains
4. Computes PD corrections (opposite to tilt)
5. Maps corrections to action space
6. Applies speed scaling (prefall only)
7. Blends with baseline using `edon_gain`
8. Safety clamps relative to baseline
9. Final clip to [-1, 1]

### 5. Key Design Principles

1. **Always stabilize opposite to tilt**: If roll > 0 (falling right), correction pushes left (negative)
2. **State-aware aggression**: OVERLOAD/STRESS in prefall get 2.0-2.5x gains, BALANCED gets 1.3x
3. **Safe zone minimalism**: In safe zone, EDON mostly tracks baseline (Kp=0.5-1.1)
4. **Prefall focus**: EDON's biggest impact is in prefall zone when stressed
5. **Safety clamping**: EDON corrections limited to 30% more than baseline magnitude (prefall) or 20% (safe)

## Expected Performance

Design targets (not guaranteed, but achievable with proper tuning):

- **20-30% fewer interventions** on medium_stress and high_stress
- **20-40% reduction** in prefall_events / prefall_time
- **15-25% improvement** in stability score

## Testing

```bash
# Medium stress, 30 episodes
python run_eval.py --mode baseline --episodes 30 --profile medium_stress --output results/baseline_medium_state_aware.json
python run_eval.py --mode edon --episodes 30 --profile medium_stress --edon-gain 0.75 --output results/edon_medium_state_aware.json
python plot_results.py --baseline results/baseline_medium_state_aware.json --edon results/edon_medium_state_aware.json --output plots/state_aware_medium

# High stress, 30 episodes
python run_eval.py --mode baseline --episodes 30 --profile high_stress --output results/baseline_high_state_aware.json
python run_eval.py --mode edon --episodes 30 --profile high_stress --edon-gain 0.75 --output results/edon_high_state_aware.json
python plot_results.py --baseline results/baseline_high_state_aware.json --edon results/edon_high_state_aware.json --output plots/state_aware_high
```

## Files Modified

1. **`evaluation/edon_state.py`** (new): EDON state enum and mapping function
2. **`run_eval.py`**: Complete refactor of `apply_edon_regulation()` with state-aware logic

## Next Steps

1. Run evaluations and verify improvements
2. Tune hyperparameters if needed (especially `EDON_BASE_KP_*` and state-dependent scales)
3. Consider adding CAV vector fine-tuning for additional precision
4. Monitor prefall event reduction as primary success metric

