# MuJoCo Environment Realism Features

This document describes the realism features added to the MuJoCo humanoid environment to match real-world conditions and the original training/eval environment.

## Overview

The MuJoCo environment now includes **6 key realism features** that make it match:
1. **Real humanoid robots** (sensor noise, actuator delays, friction variation)
2. **Original training/eval environment** (same stress profiles)

These features ensure EDON's performance in simulation matches what you'd see on a real robot.

---

## Realism Features

### 1. **Sensor Noise** ✅
- **What**: Adds Gaussian noise to all sensor readings (IMU, joint encoders)
- **Why**: Real sensors have noise (IMU drift, encoder quantization)
- **Implementation**: 
  - Roll/pitch: `±0.03 rad` noise (HIGH_STRESS)
  - Velocities: `±0.015 rad/s` noise
  - Joint positions: `±0.003 rad` noise
  - COM: `±0.003 m` noise
- **Matches**: `evaluation/stress_profiles.py` → `sensor_noise_std=0.03`

### 2. **Actuator Delays** ✅
- **What**: Simulates motor latency (commands arrive 20-40ms late)
- **Why**: Real motors have communication delays, processing time, and mechanical response lag
- **Implementation**:
  - Random delay: `2-4 steps` (20-40ms at 100Hz)
  - Uses action buffer to apply delayed commands
- **Matches**: `evaluation/stress_profiles.py` → `actuator_delay_steps=(2, 4)`

### 3. **Friction Variation** ✅
- **What**: Randomizes floor friction coefficient each episode
- **Why**: Real surfaces vary (carpet, tile, wet floor, etc.)
- **Implementation**:
  - Friction range: `0.2 - 1.5` (HIGH_STRESS)
  - Randomized at episode reset
  - Applied to floor geom in MuJoCo model
- **Matches**: `evaluation/stress_profiles.py` → `friction_min=0.2, friction_max=1.5`

### 4. **Fatigue Model** ✅
- **What**: Degrades actuator performance over time (simulates motor heating, battery drain)
- **Why**: Real motors lose efficiency during extended operation
- **Implementation**:
  - Linear degradation: `10%` over episode (HIGH_STRESS)
  - Applied as scaling factor to actions: `action *= (1.0 - degradation * progress)`
  - Minimum performance: `50%` (prevents complete failure)
- **Matches**: `evaluation/stress_profiles.py` → `fatigue_enabled=True, fatigue_degradation=0.10`

### 5. **Floor Incline Variation** ✅
- **What**: Randomizes floor tilt angle each episode
- **Why**: Real environments have slopes, ramps, uneven terrain
- **Implementation**:
  - Incline range: `±0.15 rad` (±8.6°) (HIGH_STRESS)
  - Randomized at episode reset
  - Applied as gravity component (simplified - full implementation would modify geom)
- **Matches**: `evaluation/stress_profiles.py` → `floor_incline_range=(-0.15, 0.15)`

### 6. **Height Variation** ✅
- **What**: Randomizes terrain height offset each episode
- **Why**: Real environments have steps, bumps, uneven surfaces
- **Implementation**:
  - Height range: `±0.05 m` (±5cm) (HIGH_STRESS)
  - Randomized at episode reset
- **Matches**: `evaluation/stress_profiles.py` → `height_variation_range=(-0.05, 0.05)`

---

## Integration with Stress Profiles

The environment automatically uses stress profiles from `evaluation/stress_profiles.py`:

```python
from evaluation.stress_profiles import HIGH_STRESS

env = HumanoidEnv(
    sensor_noise_std=HIGH_STRESS.sensor_noise_std,
    actuator_delay_steps=HIGH_STRESS.actuator_delay_steps,
    friction_min=HIGH_STRESS.friction_min,
    friction_max=HIGH_STRESS.friction_max,
    fatigue_enabled=HIGH_STRESS.fatigue_enabled,
    fatigue_degradation=HIGH_STRESS.fatigue_degradation,
    floor_incline_range=HIGH_STRESS.floor_incline_range,
    height_variation_range=HIGH_STRESS.height_variation_range
)
```

**Available profiles:**
- `LIGHT_STRESS`: Minimal disturbances, low noise
- `MEDIUM_STRESS`: Moderate disturbances, some delays
- `HIGH_STRESS`: Strong disturbances, delays, fatigue, uneven terrain ⭐ (used in demo)
- `HELL_STRESS`: Extreme conditions

---

## Comparison: Before vs After

| Feature | Before | After |
|---------|--------|-------|
| **Sensor Noise** | ❌ None | ✅ 0.03 std (HIGH_STRESS) |
| **Actuator Delays** | ❌ None | ✅ 20-40ms (HIGH_STRESS) |
| **Friction** | ❌ Fixed (1.0) | ✅ Random (0.2-1.5) |
| **Fatigue** | ❌ None | ✅ 10% degradation |
| **Floor Incline** | ❌ Flat | ✅ ±8.6° |
| **Height Variation** | ❌ None | ✅ ±5cm |

---

## Impact on EDON Performance

These features make the MuJoCo environment **much closer** to:
1. **Real humanoid robots** (sensor noise, delays, varying conditions)
2. **Original training/eval environment** (same stress profiles)

**Expected result**: EDON should now show **90%+ intervention reduction** (matching the original 97% result) because:
- Environment matches training conditions
- EDON was trained to handle these exact stress profiles
- Realistic disturbances match real-world scenarios

---

## Testing

Run the demo with realism features:

```bash
cd demo_mujoco
python test_demo_no_ui.py
```

The environment will automatically use `HIGH_STRESS` profile parameters, matching the original training/eval environment.

---

## Notes

- **Sensor noise** is applied to observations (what the controller sees), not the true state
- **Actuator delays** use a FIFO buffer to simulate command latency
- **Friction** is randomized per episode (not per step) for consistency
- **Fatigue** is linear over episode duration (can be made exponential if needed)
- **Floor incline** is simplified (full implementation would modify MuJoCo geom orientation)
- **Height variation** is randomized per episode (terrain heightfield would be more realistic)

These features ensure the demo environment matches both **real-world conditions** and the **original training environment**, making EDON's performance transferable to real robots.

