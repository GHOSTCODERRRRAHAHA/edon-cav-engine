# Quick Training Options for Demo/Call

## Fastest Option (30-45 minutes)
```powershell
python train_edon_mujoco.py --episodes 30 --max-steps 500
```
- **Time**: ~30-45 minutes
- **Good for**: Quick proof-of-concept
- **Performance**: Will show improvement, but not fully optimized

## Recommended for Demo (1-1.5 hours)
```powershell
python train_edon_mujoco.py --episodes 50 --max-steps 1000
```
- **Time**: ~1-1.5 hours
- **Good for**: Demo showing training process
- **Performance**: Decent improvement, good for showing the concept

## Balanced (2-2.5 hours)
```powershell
python train_edon_mujoco.py --episodes 100 --max-steps 500
```
- **Time**: ~2-2.5 hours
- **Good for**: More thorough demo
- **Performance**: Better improvement, more convincing

## Tips for Your Call

1. **Start training before the call** - Even 30 episodes will show improvement
2. **Show live progress** - The training shows real-time updates
3. **Point out adaptive memory** - It's learning from each intervention
4. **Compare before/after** - Show zero-shot vs trained performance

## What to Show

- **Before training**: Zero-shot performance (25-50% improvement)
- **During training**: Live progress, intervention reduction over time
- **After training**: Trained performance (90%+ improvement expected)

## Quick Demo Script

1. Show zero-shot demo first (2-3 minutes)
2. Start training in background
3. Show training progress during call
4. If time allows, show trained model performance

