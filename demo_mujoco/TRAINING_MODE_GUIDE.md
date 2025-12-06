# EDON Training Mode Guide

## Overview

The demo now supports two modes:

1. **Zero-Shot Mode** (default): Uses EDON Core API for inference
2. **Trained Mode**: Uses a trained policy model (no API calls)

## Training EDON on MuJoCo

### Step 1: Train the Model

Use the training script that uses the same OEM API endpoints:

```powershell
cd demo_mujoco
python train_edon_mujoco.py \
  --episodes 300 \
  --max-steps 1000 \
  --lr 5e-4 \
  --output-dir models \
  --model-name edon_v8_mujoco \
  --edon-url http://localhost:8000
```

**Requirements:**
- EDON server must be running (`python -m app.main`)
- Training uses the same OEM API endpoints:
  - `POST /oem/robot/stability` (for inference)
  - `POST /oem/robot/stability/record-outcome` (for adaptive learning)

**What gets trained:**
- Policy network (strategy selection + modulations)
- Uses PPO (Proximal Policy Optimization)
- Learns from intervention outcomes

**What doesn't get trained:**
- EDON Core engine (remains unchanged, IP-protected)
- Fail-risk model (optional, can be retrained separately)

### Step 2: Run Demo with Trained Model

```powershell
python run_demo.py \
  --mode trained \
  --trained-model models/edon_v8_mujoco.pt
```

## Comparison: Zero-Shot vs Trained

| Feature | Zero-Shot | Trained |
|---------|-----------|---------|
| **API Calls** | Yes (every 10 steps) | No (local inference) |
| **Performance** | 25-50% improvement | 90%+ improvement |
| **Setup** | Just start EDON server | Train model first |
| **Speed** | Slower (API latency) | Faster (no API) |
| **Adaptability** | Uses adaptive memory | Uses learned policy |

## For OEMs

This demonstrates the **full OEM training workflow**:

1. **Training Phase**: Use `train_edon_mujoco.py` to train on your environment
   - Uses OEM API endpoints (same as production)
   - Trains policy network (not EDON Core)
   - Saves trained model

2. **Inference Phase**: Use trained model in production
   - No API calls needed (faster)
   - Better performance (90%+ improvement)
   - Same API interface as zero-shot

## Training Script Details

The `train_edon_mujoco.py` script:

- Uses `MuJoCoTrainingEnv` wrapper around `HumanoidEnv`
- Calls `POST /oem/robot/stability` for EDON inference
- Records outcomes via `POST /oem/robot/stability/record-outcome`
- Trains policy network using PPO
- Saves trained model to `models/edon_v8_mujoco.pt`

This is the **exact same workflow** OEMs will use in production.

