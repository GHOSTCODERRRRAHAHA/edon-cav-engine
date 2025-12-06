# Training Status

## Current Training Run

**Model**: `edon_v8_strategy_temporal_v1.pt`
**Total Episodes**: 300
**Configuration**: Temporal memory (248 input size) + early-warning features

## Training Command Running

```bash
python training/train_edon_v8_strategy.py \
  --episodes 300 \
  --profile high_stress \
  --seed 0 \
  --lr 5e-4 \
  --gamma 0.995 \
  --update-epochs 10 \
  --output-dir models \
  --model-name edon_v8_strategy_temporal_v1 \
  --fail-risk-model models/edon_fail_risk_v1_fixed_v2.pt \
  --max-steps 1000 \
  --w-intervention 20.0 \
  --w-stability 1.0 \
  --w-torque 0.1 \
  --retroactive-steps 20 \
  --w-retroactive 3.0
```

## Check Progress

Run this to see current progress:
```bash
python scripts/check_training_progress.py
```

## See Live Updates

Run this for real-time monitoring:
```bash
python scripts/monitor_training_live_temporal.py
```

## Expected Output

When training is running, you should see:
- `[V8] Input size with stacked observations (8 frames): 248`
- Episode-by-episode progress: `[V8] ep=1/300 score=... reward=...`
- Reward breakdowns every 10 episodes: `R_int=... R_stab=... R_retro=...`
- Summary every 10 episodes: `[V8-SUMMARY] Episodes X-Y: ...`
- Model saves: `[EDON-V8] Saving model to models\edon_v8_strategy_temporal_v1.pt`

Training is running in the background. Check progress with the scripts above!

