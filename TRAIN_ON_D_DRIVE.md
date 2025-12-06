# Training v8 Policy on D Drive

Yes! You can train the policy on the D drive. The training script already supports custom output directories.

## Quick Command

```bash
python training/train_edon_v8_strategy.py \
  --episodes 300 \
  --profile high_stress \
  --seed 0 \
  --lr 5e-4 \
  --gamma 0.995 \
  --update-epochs 10 \
  --output-dir "D:/edon_models" \
  --model-name edon_v8_strategy_v1_trained \
  --fail-risk-model models/edon_fail_risk_v1_fixed.pt \
  --max-steps 1000
```

## What This Does

- **Saves model to**: `D:/edon_models/edon_v8_strategy_v1_trained.pt`
- **Creates directory**: Automatically creates `D:/edon_models/` if it doesn't exist
- **Training data**: Still uses the same training process, just saves to D drive

## Using the Batch Script

I've created a batch script for convenience:

**Windows:**
```bash
scripts/train_v8_on_d_drive.bat
```

**Linux/Mac:**
```bash
bash scripts/train_v8_on_d_drive.sh
```

## Notes

1. **Fail-risk model**: Still loads from `models/edon_fail_risk_v1_fixed.pt` (C drive)
   - You can copy it to D drive if needed: `copy models/edon_fail_risk_v1_fixed.pt D:/edon_models/`
   - Then use: `--fail-risk-model "D:/edon_models/edon_fail_risk_v1_fixed.pt"`

2. **Evaluation**: When evaluating, you'll need to specify the D drive model path:
   ```bash
   python run_eval.py \
     --mode edon \
     --profile high_stress \
     --episodes 30 \
     --seed 42 \
     --output results/edon_v8_drive.json \
     --edon-gain 1.0 \
     --edon-arch v8_strategy \
     --edon-score
   ```
   (You may need to update `run_eval.py` to look for models in D drive, or copy the model back to `models/`)

3. **Disk space**: Training 300 episodes will create a model file (~1-5 MB), so D drive is fine for storage.

## Alternative: Copy After Training

If you prefer to train on C drive but move to D drive after:

```bash
# Train on C drive (default)
python training/train_edon_v8_strategy.py --episodes 300 ...

# Copy to D drive
copy models/edon_v8_strategy_v1_trained.pt D:/edon_models/
```

