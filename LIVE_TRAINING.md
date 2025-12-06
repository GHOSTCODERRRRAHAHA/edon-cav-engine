# Live Training Monitor

## Training Started

The EDON v8 Temporal Memory policy training is running with:

- **Model**: `edon_v8_strategy_temporal_v1.pt`
- **Episodes**: 300
- **Input Size**: 248 (31 base features × 8 stacked frames)
- **New Features**: Early-warning + temporal memory

## To See Live Output

Run this command in your terminal:

```bash
python scripts/monitor_training_live_temporal.py
```

This will show:
- Real-time episode updates
- Score, reward, length, interventions
- Summary every 10 episodes
- Model checkpoint updates

## Or Check Latest Output

```bash
python scripts/show_training_output.py
```

## Training Command (for reference)

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

## What to Watch For

- **Input size**: Should show 248 (confirming stacked observations)
- **Episode progress**: Episodes 1, 10, 50, 100, 150, 200, 250, 300
- **Reward breakdown**: R_int, R_stab, R_torque, R_retro (when available)
- **Interventions**: Should decrease over time
- **Stability**: Should stay reasonable (not explode)

Training is running in the background. Use the monitor scripts to see live progress!

