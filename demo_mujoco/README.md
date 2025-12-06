# EDON MuJoCo Stability Demo

OEM-ready side-by-side comparison demo showing baseline vs EDON-stabilized humanoid control.

## Overview

This demo provides a **clean, technical comparison** of humanoid stability control:
- **Baseline**: Standard PD controller for balance
- **EDON-Stabilized**: Baseline + EDON stabilization layer

Both controllers face **identical disturbances** (pushes, terrain, load shifts, latency jitter) for fair comparison.

## Quick Start

### Prerequisites

1. **Install MuJoCo:**
   ```bash
   pip install mujoco>=3.1.0
   ```

2. **Install other dependencies:**
   ```bash
   cd demo_mujoco
   pip install -r requirements.txt
   ```

3. **Ensure EDON server is running:**
   ```bash
   # In another terminal, from project root
   python -m app.main
   # Or use your existing EDON server at http://localhost:8000
   ```

### Running the Demo

**Option 1: PowerShell (Windows) - Recommended**
```powershell
.\start_demo.ps1
```
This automatically disables adaptive memory for consistent performance.

**Option 2: Python directly**
```bash
python run_demo.py
```
Adaptive memory is disabled by default for consistent demo performance.

**Option 3: With custom options**
```bash
python run_demo.py --port 8080 --edon-url http://localhost:8000 --duration 30.0
```

**Note**: Adaptive memory is **disabled by default** for consistent demo performance. 
To enable adaptive learning, set `EDON_DISABLE_ADAPTIVE_MEMORY=0` before running.

### Accessing the UI

1. Open your browser: `http://localhost:8080`
2. Click **"Start Demo"** to begin the comparison
3. Watch real-time metrics update side-by-side

## Features

### Side-by-Side Comparison
- **Baseline Controller**: Standard PD control for balance
- **EDON-Stabilized**: Baseline + EDON modulation layer
- Both run in parallel with identical disturbances

### Real-Time Metrics
- **Falls**: Number of falls per episode
- **Freezes**: Periods of low motion
- **Interventions**: Corrective actions needed
- **Recovery Time**: Time to recover from disturbances
- **Stability Score**: Overall stability metric

### Disturbances (Deterministic & Replayable)
- **Impulse Pushes**: Lateral, frontal, and diagonal forces
- **Uneven Terrain**: Heightfield bumps
- **Dynamic Load Shifts**: Moving center of mass
- **Latency Jitter**: Random control delays (10-50ms)

### Safety Controls
- **EDON ON/OFF**: Toggle EDON stabilization
- **Kill Switch**: Emergency stop
- **Start/Stop**: Control demo execution

## Architecture

```
demo_mujoco/
├── sim/
│   ├── env.py              # MuJoCo environment wrapper
│   └── simple_humanoid.xml # Humanoid robot model
├── controllers/
│   ├── baseline_controller.py  # PD balance controller
│   └── edon_layer.py           # EDON stabilization layer
├── disturbances/
│   └── generator.py        # Deterministic disturbance generation
├── metrics/
│   └── tracker.py          # Metrics computation
├── ui/
│   ├── server.py           # FastAPI backend
│   └── index.html          # Web UI frontend
├── run_demo.py             # Main demo orchestrator
└── requirements.txt        # Dependencies
```

## How It Works

1. **Disturbance Generation**: Creates a deterministic script of disturbances
2. **Parallel Execution**: Runs baseline and EDON episodes simultaneously
3. **State Tracking**: Monitors robot state (roll, pitch, COM, etc.)
4. **EDON Integration**: Calls `/oem/robot/stability` API for control modulations
5. **Metrics Computation**: Tracks falls, freezes, interventions, recovery times
6. **Live Updates**: WebSocket updates UI in real-time

## API Integration

The EDON layer calls:
```
POST /oem/robot/stability
{
  "robot_state": {
    "roll": 0.05,
    "pitch": 0.02,
    "roll_velocity": 0.1,
    "pitch_velocity": 0.05,
    "com_x": 0.0,
    "com_y": 0.0
  },
  "history": [...]  # Optional: last 8 states
}
```

And receives control modulations:
- `gain_scale`: Action scaling factor
- `compliance`: Bias application strength
- `bias`: Corrective torque adjustments

## Customization

### Adjust Controller Gains
Edit `controllers/baseline_controller.py`:
```python
controller = BaselineController(
    kp_roll=100.0,
    kd_roll=10.0,
    # ... adjust gains
)
```

### Modify Disturbances
Edit `disturbances/generator.py`:
```python
script = generator.generate_script(
    push_probability=0.15,  # More/less frequent pushes
    terrain_bumps=5,       # Number of terrain changes
    # ... adjust parameters
)
```

### Change Episode Duration
```bash
python run_demo.py --duration 60.0  # 60 second episodes
```

## Troubleshooting

**EDON server not found:**
- Ensure EDON server is running at `http://localhost:8000`
- Check with: `curl http://localhost:8000/health`

**MuJoCo import errors:**
- Install MuJoCo: `pip install mujoco>=3.1.0`
- On Linux, may need: `sudo apt-get install libgl1-mesa-glx`

**UI not loading / Port already in use:**
- The demo will automatically try the next available port if 8080 is taken
- Or specify a different port: `--port 8081`
- Check what's using the port: `netstat -ano | findstr :8080` (Windows)

**Simulation too slow:**
- Reduce episode duration: `--duration 15.0`
- Disable rendering (already disabled by default)

## Notes

- This is a **technical demo**, not a research UI
- Designed for **OEM screen-sharing** and presentations
- Disturbances are **deterministic** for fair comparison
- Both controllers use the **same seed** and **same disturbance script**

## License

Part of the EDON project. See main project license.

