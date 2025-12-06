# EDON Humanoid Evaluation - Quick Start

## What This Does

This evaluation system lets you run A/B tests comparing:
- **Baseline**: Your control policy without EDON
- **EDON**: Same policy with EDON feeding state into control logic

## Installation

1. **Install dependencies:**
   ```bash
   pip install numpy matplotlib
   pip install edon-2.0.0-py3-none-any.whl  # EDON SDK
   ```

2. **Start EDON server** (if not already running):
   ```bash
   # Option 1: Docker
   docker run -p 8001:8001 edon-server:v2.0.0
   
   # Option 2: Direct Python
   export EDON_MODE=v2
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
   ```

## Quick Test (Mock Environment)

Run with the included mock environment to test the system:

```bash
# 1. Run baseline
python run_eval.py --mode baseline --episodes 10 --output results/baseline.json

# 2. Run EDON
python run_eval.py --mode edon --episodes 10 --output results/edon.json

# 3. Plot results
python plot_results.py --baseline results/baseline.json --edon results/edon.json
```

## Integration with Your Simulator

### 1. Replace Environment

Edit `run_eval.py`, find `make_humanoid_env()` and replace:

```python
def make_humanoid_env(seed: Optional[int] = None):
    # Your environment here
    import gym
    env = gym.make("Humanoid-v3")
    if seed is not None:
        env.seed(seed)
    return env
```

### 2. Update Observation Extraction

Edit `evaluation/humanoid_runner.py`, find `_extract_stability_metrics()` and adapt to your observation format.

### 3. Update Controllers

Edit `run_eval.py`, replace `baseline_controller()` and `edon_controller()` with your actual control policies.

### 4. Update Sensor Window

Edit `evaluation/humanoid_runner.py`, find `_build_sensor_window()` and adapt to extract actual sensor data.

See `evaluation/example_integration.py` for detailed examples.

## Full Workflow

```bash
# 1. Run baseline evaluation (50 episodes, randomized environment)
python run_eval.py \
    --mode baseline \
    --episodes 50 \
    --seed 42 \
    --randomize-env \
    --output results/baseline.json

# 2. Run EDON evaluation (same seed for fair comparison)
python run_eval.py \
    --mode edon \
    --episodes 50 \
    --seed 42 \
    --randomize-env \
    --output results/edon.json \
    --edon-url http://127.0.0.1:8001

# 3. Generate comparison plots
python plot_results.py \
    --baseline results/baseline.json \
    --edon results/edon.json \
    --output plots

# 4. View results
# - Check plots/comparison.png for bar charts
# - Check plots/stability_timeseries.png for time-series
# - Check results/*.json for detailed metrics
# - Check results/*.csv for per-episode data
```

## Metrics Explained

- **Interventions**: Safety interventions (falls, torque violations, stuck states)
- **Freeze Events**: Periods where robot is stuck (no movement > 3 seconds)
- **Stability Score**: Variance of torso roll + pitch (lower = more stable)
- **Success Rate**: Percentage of episodes that complete successfully

## Output Files

- `results/baseline_results.json` - Baseline metrics
- `results/edon_results.json` - EDON metrics
- `results/*.csv` - Per-episode detailed data
- `plots/comparison.png` - Bar chart comparison
- `plots/stability_timeseries.png` - Stability over time

## Configuration

Edit `evaluation/config.py` to adjust:
- Freeze detection thresholds
- Intervention detection limits
- Environment randomization
- EDON connection settings

## Troubleshooting

**EDON connection failed:**
- Check server is running: `curl http://127.0.0.1:8001/health`
- Verify `--edon-url` matches your server

**Environment errors:**
- Replace `make_humanoid_env()` with your actual environment
- Ensure environment has `reset()`, `step()`, `render()` methods

**Observation format issues:**
- Update `_extract_stability_metrics()` to match your observation space
- Check that observation contains roll, pitch, and center of mass data

## Next Steps

1. Run with mock environment to verify setup
2. Integrate your actual simulator
3. Run full evaluation (50+ episodes recommended)
4. Analyze results and iterate on control policy

For detailed integration examples, see `evaluation/example_integration.py`.

