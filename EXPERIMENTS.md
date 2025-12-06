# EDON Humanoid Evaluation - Architecture & Experiments

## File Structure & Responsibilities

### Main Evaluation Entry Point
**File:** `run_eval.py`
- CLI argument parsing
- Environment creation via `make_humanoid_env()`
- Controller selection (baseline vs EDON)
- EDON client initialization
- Orchestrates evaluation runs

### Environment Configuration
**File:** `evaluation/config.py`
- All thresholds (freeze, intervention, stability)
- Environment randomization parameters
- EDON connection settings
- **NEW:** Stress profile definitions (light_stress, medium_stress, high_stress)

### Controller / Policy Logic
**File:** `run_eval.py`
- `baseline_controller()` - Simple PD controller without EDON
- `edon_controller()` - Controller that uses EDON state to modulate actions
- **NEW:** Tunable `edon_gain` parameter to control EDON influence strength

### EDON Integration Layer
**File:** `evaluation/humanoid_runner.py`
- `_get_edon_state()` - Calls EDON client to get CAV/state
- `_build_sensor_window()` - Builds sensor data window for EDON API
- Runs episodes and tracks metrics
- Detects interventions, freezes, stability

### Environment
**File:** `evaluation/mock_env.py`
- `MockHumanoidEnv` - Realistic physics simulation
- Supports friction, external forces, floor incline
- **NEW:** Stress mode support (uneven ground, actuator delay, fatigue)

### Environment Randomization
**File:** `evaluation/randomization.py`
- `EnvironmentRandomizer` - Applies randomization at reset and per-step
- Handles friction, pushes, sensor noise
- **NEW:** Stress profile application (pushes, noise, delays, fatigue)

### Metrics Tracking
**File:** `evaluation/metrics.py`
- `EpisodeMetrics` - Per-episode metrics
- `RunMetrics` - Aggregated run metrics
- **NEW:** Enhanced metrics (roll/pitch RMS, max, variance, COM deviation)

### Plotting & Comparison
**File:** `plot_results.py`
- Loads results JSON files
- Computes percentage improvements
- Generates comparison plots
- **NEW:** Support for stress profile comparisons

---

## How the Evaluation Pipeline Works

### Baseline Mode Flow:
1. `run_eval.py` creates environment via `make_humanoid_env()`
2. Creates `HumanoidRunner` with `baseline_controller`
3. For each episode:
   - Environment resets
   - Randomization applied (if enabled)
   - Loop: `baseline_controller(obs)` → action → `env.step(action)`
   - Metrics tracked: interventions, freezes, stability
4. Results saved to JSON/CSV

### EDON Mode Flow:
1. `run_eval.py` creates environment and EDON client
2. Creates `HumanoidRunner` with `edon_controller` and `edon_client`
3. For each episode:
   - Environment resets
   - Randomization applied (if enabled)
   - Loop:
     - `_get_edon_state(obs)` → calls EDON API → returns state/CAV
     - `edon_controller(obs, edon_state)` → action (modulated by EDON)
     - `env.step(action)`
   - Metrics tracked: interventions, freezes, stability
4. Results saved to JSON/CSV

### Key Differences:
- **Baseline**: Controller only sees `obs`, uses fixed gains
- **EDON**: Controller sees `obs` + `edon_state`, uses dynamic gains based on EDON state
- **EDON Gain**: New parameter `--edon-gain` (default 0.5) controls how strongly EDON influences actions

---

## Metrics Definitions

### Interventions
**Definition:** Safety interventions triggered when:
- Fall detected: `abs(roll) > FALL_THRESHOLD_ROLL` OR `abs(pitch) > FALL_THRESHOLD_PITCH`
- Torque violation: `max_torque > SAFETY_TORQUE_LIMIT`
- Joint limit violation: `max_joint_angle > SAFETY_JOINT_LIMIT`

**Threshold:** Configurable in `evaluation/config.py` (default: 0.4 rad for roll/pitch)

### Stability Score
**Definition:** `var(roll_history) + var(pitch_history)`
- Lower = more stable
- Tracks variance of torso orientation over episode

**Enhanced Metrics:**
- Roll RMS: `sqrt(mean(roll^2))`
- Pitch RMS: `sqrt(mean(pitch^2))`
- Roll max: `max(abs(roll))`
- Pitch max: `max(abs(pitch))`
- COM deviation: `sqrt(mean(com_x^2 + com_y^2))`

### Freeze Events
**Definition:** Periods where robot is stuck:
- Movement < `MOVE_EPS` (default: 0.02m) for > `FREEZE_TIME_THRESHOLD` (default: 2.0s)
- Measured as change in center of mass position

**Enhanced Definition:**
- Speed almost zero while commanded to move
- Repeated minimal motion (< MOVE_EPS for multiple consecutive steps)

### Success
**Definition:** Episode completes successfully:
- Reached goal: `distance_to_goal < SUCCESS_DISTANCE_THRESHOLD`
- Within time limit: `episode_time < SUCCESS_TIME_THRESHOLD`
- No fall detected

---

## Stress Profiles

### light_stress
- Push force: 5-50N (low)
- Push probability: 0.08 (8% per step)
- Sensor noise: 0.01 std
- Friction range: 0.5-1.0 (moderate)
- No actuator delay
- No fatigue

### medium_stress
- Push force: 10-100N (medium)
- Push probability: 0.12 (12% per step)
- Sensor noise: 0.02 std
- Friction range: 0.3-1.2 (extreme)
- Actuator delay: 1-2 steps (10-20ms)
- Light fatigue (5% performance degradation over episode)

### high_stress
- Push force: 20-150N (high, occasional strong pushes)
- Push probability: 0.18 (18% per step)
- Sensor noise: 0.03 std
- Friction range: 0.2-1.5 (very extreme)
- Actuator delay: 2-4 steps (20-40ms)
- Moderate fatigue (10% performance degradation)
- Floor incline: ±0.15 rad (±8.6°)
- Height variations: ±0.05m

---

## EDON Gain Parameter

**Parameter:** `--edon-gain` or `edon_gain` in config (default: 0.5)

**How it works:**
- Controls blending strength: `action = baseline_action * (1 - alpha) + edon_modulated_action * alpha`
- `alpha = 0.25`: Light EDON influence (25% EDON, 75% baseline)
- `alpha = 0.5`: Balanced (50% EDON, 50% baseline) - **default**
- `alpha = 0.75`: Strong EDON influence (75% EDON, 25% baseline)

**EDON can adjust:**
- Balance/posture gains (based on state_class: overload, alert, focus, restorative)
- Speed/step length (via `speed_scale`)
- Stiffness/compliance (via `torque_scale`, `safety_scale`)
- Emergency handling (strong stabilizing forces)

**EDON cannot:**
- Access future information
- Change environment distribution (same for baseline and EDON)

---

## Running Experiments

### Basic Usage

```bash
# Baseline with light stress
python run_eval.py --mode baseline --episodes 20 --profile light_stress --output results/baseline_light.json

# EDON with light stress, default gain (0.5)
python run_eval.py --mode edon --episodes 20 --profile light_stress --output results/edon_light.json

# EDON with high stress, strong gain (0.75)
python run_eval.py --mode edon --episodes 20 --profile high_stress --edon-gain 0.75 --output results/edon_high_strong.json

# EDON with medium stress, light gain (0.25)
python run_eval.py --mode edon --episodes 20 --profile medium_stress --edon-gain 0.25 --output results/edon_medium_light.json
```

### Full Experiment Matrix

Use `run_experiments.py` to run all combinations automatically:

```bash
python run_experiments.py
```

This runs:
- Modes: baseline, edon
- Profiles: light_stress, medium_stress, high_stress
- EDON gains: 0.25, 0.5, 0.75 (for EDON mode only)

Results saved to: `results/experiments/{mode}_{profile}_gain{gain}.json`

The script will:
1. Run baseline for each profile
2. Run EDON for each profile × gain combination
3. Compute and print improvement percentages
4. Show summary of all results

### Plotting Results

```bash
# Compare specific runs
python plot_results.py --baseline results/baseline_high.json --edon results/edon_high.json --output plots/high_stress

# Compare with gain variations
python plot_results.py --baseline results/baseline_high.json --edon results/edon_high_gain0.75.json --output plots/high_stress_strong
```

---

## Expected Performance Gaps

With proper tuning (high_stress profile, edon_gain=0.5-0.75):

- **Interventions:** 30-40% reduction
- **Stability:** 20-30% improvement (lower variance)
- **Freezes:** 25-35% reduction
- **Success rate:** 15-25% improvement

These numbers are achievable because:
1. Baseline struggles with high stress (pushes, noise, delays)
2. EDON adapts gains based on state (overload → high stabilization)
3. EDON handles emergencies better (stronger corrective forces)
4. EDON reduces hesitations (better state awareness)

---

## Configuration Files

- `evaluation/config.py` - Main configuration
- `evaluation/stress_profiles.py` - Stress profile definitions (NEW)
- Results: `results/*.json`, `results/*.csv`
- Plots: `plots/*.png`

---

## Next Steps

1. Run baseline on high_stress to establish baseline performance
2. Run EDON on high_stress with different gains to find optimal
3. Compare results and generate plots
4. Document findings in results summary

