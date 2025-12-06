# EDON Stabilization Fix - Simple Proportional Control

## Problem

The previous aggressive prefall logic (3x boost, 70% movement reduction, stiffness increases) was **destabilizing** the humanoid instead of stabilizing it.

## Solution

Replaced complex multi-step adjustment logic with **simple proportional control** based only on tilt.

## Implementation

### Core Logic (`apply_edon_regulation()` in `run_eval.py`)

```python
# 1. Compute tilt magnitude
tilt_magnitude = sqrt(roll**2 + pitch**2)

# 2. Proportional correction (opposite to tilt)
correction_roll = -Kp_roll * roll
correction_pitch = -Kp_pitch * pitch

# 3. Blend with baseline
final_action = baseline_action + edon_gain * correction
```

### Key Changes

1. **Single correction vector** - Only proportional to tilt, no complex state-based logic
2. **Small proportional gains**:
   - Safe zone: Kp = 0.05
   - Normal: Kp = 0.10
   - Prefall zone: Kp = 0.12 (slightly higher, but still modest)
3. **No aggressive multipliers**:
   - ❌ Removed 3x stabilization boost
   - ❌ Removed 70% movement reduction
   - ❌ Removed stiffness increases
   - ❌ Removed global damping (only if tilt_rate > 0.15)
4. **Modest speed scaling** (ONLY in prefall):
   - `speed_scale = lerp(1.0, 0.6, edon_gain)`
   - Applied only to forward velocity component (indices 4+)
   - When edon_gain=0.75: speed_scale = 0.7 (reduce to 70%)
5. **Simple blending**: `final_action = baseline_action + edon_gain * correction`
   - Keeps final_action close to baseline
   - Corrections are small and continuous

### Proportional Gains

- **Kp_roll = 0.10** (default)
- **Kp_pitch = 0.10** (default)
- **Prefall zone**: Kp = 0.12 (20% increase)
- **Safe zone**: Kp = 0.05 (50% reduction)
- **Safety scale boost**: Max 20% (from EDON influences)

### Velocity Damping

Only applied if `tilt_rate > 0.15` (oscillations detected):
- Kd = 0.08 (small damping gain)
- `correction[0] += -Kd * roll_velocity`
- `correction[1] += -Kd * pitch_velocity`

### Speed Scaling (Prefall Only)

```python
if tilt_zone == "prefall" and action_size > 4:
    speed_scale = 1.0 - (1.0 - 0.6) * edon_gain
    # lerp(1.0, 0.6, edon_gain)
    # edon_gain=0.0 → speed_scale=1.0 (no change)
    # edon_gain=0.75 → speed_scale=0.7 (reduce to 70%)
    # edon_gain=1.0 → speed_scale=0.6 (reduce to 60%)
    correction[4:] = baseline_action[4:] * (speed_scale - 1.0)
```

## Expected Results

With simple proportional control:

- **EDON gain 0.75 should NOT increase interventions** (stabilizing, not destabilizing)
- **10-20% reduction in prefall events** (small corrections help recovery)
- **10-20% reduction in interventions** on medium_stress (prevent progression to fail)

## Key Differences from Previous Implementation

| Previous | New |
|----------|-----|
| 3x stabilization boost | Kp = 0.12 (modest) |
| 70% movement reduction | Speed scale to 0.6-0.7 (modest) |
| Stiffness increase (1.5x) | No stiffness changes |
| Global damping | Only if tilt_rate > 0.15 |
| Complex state-based logic | Simple proportional control |
| `final = baseline * (1-gain) + regulated * gain` | `final = baseline + gain * correction` |

## Testing

```bash
# Run baseline
python run_eval.py --mode baseline --episodes 30 --profile medium_stress --output results/baseline_medium.json

# Run EDON with gain 0.75
python run_eval.py --mode edon --episodes 30 --profile medium_stress --edon-gain 0.75 --output results/edon_medium.json

# Compare - should see:
# - Interventions: EDON <= Baseline (not higher)
# - Prefall events: 10-20% reduction
# - Stability: 10-20% improvement
```

## Files Modified

- `run_eval.py` - Completely rewrote `apply_edon_regulation()` to use simple proportional control

The new implementation is **stabilizing** (small corrections) rather than **destabilizing** (aggressive overcorrections).

