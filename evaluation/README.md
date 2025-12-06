# EDON Humanoid Evaluation System

A comprehensive evaluation framework for benchmarking EDON's impact on humanoid robot autonomy in simulation.

## Overview

This package provides tools to run A/B tests comparing:
- **Baseline**: Control policy without EDON
- **EDON**: Same policy with EDON feeding state into control logic

## Quick Start

### 1. Run Baseline Evaluation

```bash
python run_eval.py --mode baseline --episodes 50 --output results/baseline.json --randomize-env
```

### 2. Run EDON Evaluation

```bash
python run_eval.py --mode edon --episodes 50 --output results/edon.json --randomize-env
```

### 3. Plot Comparison

```bash
python plot_results.py --baseline results/baseline.json --edon results/edon.json --output plots
```

## Metrics Tracked

1. **Interventions**: Count of safety interventions (falls, torque violations, stuck states)
2. **Freeze Events**: Count of periods where robot is stuck (no movement > threshold)
3. **Stability Score**: Variance of torso roll + pitch (lower = better)
4. **Episode Length**: Number of steps per episode
5. **Success Rate**: Percentage of episodes that complete successfully

## Integration with Your Simulator

### Step 1: Replace Mock Environment

Edit `run_eval.py` and replace the `make_humanoid_env()` function:

```python
def make_humanoid_env(seed: Optional[int] = None):
    # Replace with your environment
    import gym
    env = gym.make("Humanoid-v3")
    if seed is not None:
        env.seed(seed)
    return env
```

### Step 2: Adapt Observation Extraction

Edit `evaluation/humanoid_runner.py` and update `_extract_stability_metrics()` to match your observation space:

```python
def _extract_stability_metrics(self, obs: Dict[str, Any]) -> tuple[float, float, float]:
    # Adapt to your observation format
    roll = obs["torso_roll"]  # or obs[0], obs["roll"], etc.
    pitch = obs["torso_pitch"]
    com = np.sqrt(obs["com_x"]**2 + obs["com_y"]**2)
    return roll, pitch, com
```

### Step 3: Update Controller

Replace `baseline_controller()` and `edon_controller()` in `run_eval.py` with your actual control policies.

### Step 4: Update Sensor Window Building

Edit `evaluation/humanoid_runner.py` and update `_build_sensor_window()` to extract actual sensor data from your robot:

```python
def _build_sensor_window(self, obs: Dict[str, Any]) -> Dict[str, Any]:
    # Extract actual sensor readings (240 samples @ 60Hz)
    # In real-time, you'd buffer these from actual sensors
    window = {
        "physio": {
            "EDA": self.eda_buffer[-240:],  # Your actual EDA sensor buffer
            "BVP": self.bvp_buffer[-240:]
        },
        "motion": {
            "ACC_x": self.acc_x_buffer[-240:],
            "ACC_y": self.acc_y_buffer[-240:],
            "ACC_z": self.acc_z_buffer[-240:]
        },
        # ... etc
    }
    return window
```

## Configuration

Edit `evaluation/config.py` to adjust:
- Freeze detection thresholds
- Stability computation parameters
- Intervention detection limits
- Environment randomization settings
- EDON connection parameters

## Output Files

### JSON Results
- `results/baseline_results.json`: Aggregated metrics for baseline run
- `results/edon_results.json`: Aggregated metrics for EDON run

### CSV Logs
- `results/baseline_results.csv`: Per-episode metrics (if enabled)
- `results/edon_results.csv`: Per-episode metrics (if enabled)

### Plots
- `plots/comparison.png`: Bar charts comparing metrics
- `plots/stability_timeseries.png`: Time-series stability plots

## Command-Line Options

### run_eval.py

```
--mode {baseline,edon}    Evaluation mode (required)
--episodes N              Number of episodes (default: 50)
--output PATH              Output JSON path (default: results/{mode}_results.json)
--seed N                   Random seed for reproducibility
--randomize-env            Enable environment randomization
--edon-url URL             EDON server URL (default: http://127.0.0.1:8001)
--edon-grpc                Use gRPC transport for EDON
--render                   Render episodes (slower)
```

### plot_results.py

```
--baseline PATH            Path to baseline results JSON (required)
--edon PATH                Path to EDON results JSON (required)
--output DIR               Output directory for plots (default: plots/)
```

## Example Workflow

```bash
# 1. Start EDON server (if not already running)
# docker run -p 8001:8001 edon-server:v2.0.0

# 2. Run baseline evaluation
python run_eval.py \
    --mode baseline \
    --episodes 50 \
    --seed 42 \
    --randomize-env \
    --output results/baseline.json

# 3. Run EDON evaluation
python run_eval.py \
    --mode edon \
    --episodes 50 \
    --seed 42 \
    --randomize-env \
    --output results/edon.json \
    --edon-url http://127.0.0.1:8001

# 4. Generate comparison plots
python plot_results.py \
    --baseline results/baseline.json \
    --edon results/edon.json \
    --output plots

# 5. View results
# - Check plots/ directory for visualizations
# - Check results/ directory for JSON/CSV data
```

## Dependencies

- Python 3.10+
- numpy
- matplotlib
- edon SDK (install from wheel: `pip install edon-2.0.0-py3-none-any.whl`)

## Troubleshooting

### EDON Connection Failed
- Make sure EDON server is running: `curl http://127.0.0.1:8001/health`
- Check `--edon-url` matches your server address
- For gRPC, ensure port 50052 is accessible

### Environment Not Found
- Replace `make_humanoid_env()` with your actual environment creation
- Ensure environment has `reset()`, `step()`, and `render()` methods

### Observation Format Mismatch
- Update `_extract_stability_metrics()` to match your observation space
- Check that observation contains `roll`, `pitch`, and center of mass data

### No Plots Generated
- Ensure both baseline and EDON results exist
- Check that results JSON files are valid
- Verify matplotlib is installed: `pip install matplotlib`

## License

See main project license.

