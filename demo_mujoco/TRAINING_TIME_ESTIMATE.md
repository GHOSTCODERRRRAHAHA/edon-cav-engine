# Training Time Estimate

## Current Configuration
- **Episodes**: 300
- **Steps per episode**: 1000 (default)
- **Simulation timestep**: 0.01s (100Hz)
- **API calls**: Every step (1000 calls per episode)
- **API latency**: ~0.1-0.2s per call (average ~0.15s)

## Time Breakdown Per Episode

1. **Simulation time**: 1000 steps × 0.01s = **10 seconds**
2. **API calls**: 1000 calls × 0.15s = **150 seconds** (2.5 minutes)
3. **PPO update**: ~1 second
4. **Total per episode**: ~**161 seconds** (~2.7 minutes)

## Total Training Time

**300 episodes × 2.7 minutes = ~810 minutes = ~13.5 hours**

## Optimization Options

### Option 1: Reduce Episodes (Quick Test)
```powershell
python train_edon_mujoco.py --episodes 50 --max-steps 1000
```
**Time**: ~2.25 hours

### Option 2: Reduce Steps Per Episode
```powershell
python train_edon_mujoco.py --episodes 300 --max-steps 500
```
**Time**: ~6.75 hours

### Option 3: Both (Fast Training)
```powershell
python train_edon_mujoco.py --episodes 100 --max-steps 500
```
**Time**: ~2.25 hours

### Option 4: Full Training (Recommended for Production)
```powershell
python train_edon_mujoco.py --episodes 300 --max-steps 1000
```
**Time**: ~13.5 hours (overnight)

## Notes

- The API latency is the main bottleneck (150s vs 10s simulation)
- Adaptive memory will learn throughout training
- Checkpoints are saved every 50 episodes
- You can stop and resume training using checkpoints

## Progress Tracking

The training shows live updates after each episode, so you can monitor progress:
- Episode completion time
- Intervention counts
- Reward trends
- Loss values

