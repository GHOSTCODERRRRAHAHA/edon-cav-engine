# Running Zero-Shot Demo While Training

## ✅ Yes, You Can Run Both Simultaneously!

The zero-shot demo and training both use the same EDON server, so you can run them at the same time.

## Setup

### Terminal 1: EDON Server (Already Running)
```powershell
python -m app.main
```
- This is already running for your training
- **Don't stop it** - both demo and training need it

### Terminal 2: Training (Already Running)
```powershell
cd demo_mujoco
python train_edon_mujoco.py --episodes 50 --max-steps 1000
```
- This is already running
- **Leave it running** - it will continue training in the background

### Terminal 3: Zero-Shot Demo (New Terminal)
```powershell
cd demo_mujoco
python run_demo.py
```
- Open a **new terminal/PowerShell window**
- Run the demo here
- It will use the same EDON server

## What Happens

1. **EDON Server**: Handles requests from both demo and training
2. **Training**: Continues in background, making API calls
3. **Demo**: Runs zero-shot comparison, also making API calls
4. **Both work**: The server handles both simultaneously

## Performance

- **Demo**: May be slightly slower due to training API calls
- **Training**: May be slightly slower due to demo API calls
- **Both still work**: The server can handle both

## For Your Call

1. **Start training** (already done)
2. **Open demo in new terminal** - Show zero-shot performance
3. **Switch back to training terminal** - Show training progress
4. **Both running**: Demonstrate that EDON can do inference while learning

## Tips

- **Demo first**: Show zero-shot performance (2-3 minutes)
- **Then show training**: Switch to training terminal to show progress
- **Both visible**: You can have both terminals visible side-by-side

## Commands Summary

```powershell
# Terminal 1: EDON Server (keep running)
python -m app.main

# Terminal 2: Training (keep running)
cd demo_mujoco
python train_edon_mujoco.py --episodes 50 --max-steps 1000

# Terminal 3: Demo (new terminal)
cd demo_mujoco
python run_demo.py
```

## Note

The demo uses adaptive memory **disabled** by default (for consistent performance), while training has adaptive memory **enabled** (for learning). They don't interfere with each other.

