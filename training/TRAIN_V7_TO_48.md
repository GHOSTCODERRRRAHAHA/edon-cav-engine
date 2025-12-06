# Quick Guide: Train v7 to Reach Score 48

## Step 1: Reward Function (Already Updated)

The reward function in `training/edon_score.py` has been optimized with:
- **2x stronger intervention penalty** (20.0 vs 10.0)
- **1.6x stronger tilt penalty** (8.0 vs 5.0)
- **1.5x stronger velocity penalty** (3.0 vs 2.0)
- **Positive rewards for stable states** (new)
- **Higher base reward** (0.2 vs 0.1)

## Step 2: Train the Model

```bash
python training/train_edon_v7.py \
  --episodes 200 \
  --profile high_stress \
  --seed 42 \
  --lr 1e-4 \
  --gamma 0.995 \
  --update-epochs 6 \
  --output-dir models \
  --model-name edon_v7_target48
```

**Key changes from default:**
- `--episodes 200` (10x more training)
- `--lr 1e-4` (more stable learning)
- `--gamma 0.995` (values future rewards more)
- `--update-epochs 6` (more refinement)

**Expected training time:** 2-4 hours

## Step 3: Evaluate

```bash
python run_eval.py \
  --mode edon \
  --profile high_stress \
  --episodes 30 \
  --seed 42 \
  --output results/edon_v7_target48.json \
  --edon-gain 1.0 \
  --edon-arch v7_learned \
  --edon-score
```

**Check the EDON Episode Score** - should be ≥ 48

## Step 4: Multi-Seed Validation (Optional)

If score ≥ 48, validate across multiple seeds:

```bash
foreach ($s in 0,1,2,3,4) {
  python run_eval.py --mode edon --profile high_stress --episodes 30 --seed $s --output results/edon_v7_target48_seed_$s.json --edon-gain 1.0 --edon-arch v7_learned --edon-score
}
```

## Expected Results

**Target Metrics:**
- **Interventions:** 41.2 → **35.0** (-6.2)
- **Stability:** 0.0199 → **0.0149** (-0.005)
- **Length:** 304.8 → **350.0** (+45.2)
- **EDON Score:** 40.3 → **48.0** (+7.7)

## If Score < 48 After Training

1. **Check training logs:** Did episode scores improve over time?
2. **If interventions still high:** Increase intervention penalty further (try 25.0 or 30.0)
3. **If stability poor:** Increase tilt penalty (try 10.0)
4. **If episodes too short:** Increase base reward (try 0.3)
5. **Retrain with adjusted rewards**

## Troubleshooting

**Problem:** Training score not improving
- **Solution:** Lower learning rate (try 5e-5) or increase update epochs (try 8)

**Problem:** Model too conservative (no action)
- **Solution:** Reduce tilt/velocity penalties slightly, or add exploration bonus

**Problem:** Episodes ending too early
- **Solution:** Increase base reward or reduce intervention penalty slightly

