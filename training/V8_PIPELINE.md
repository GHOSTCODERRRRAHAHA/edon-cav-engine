# EDON v8 End-to-End Pipeline

Complete guide for training and evaluating EDON v8.

## Step 1: Train Fail-Risk Model

Train the predictive failure model from existing JSONL logs:

```bash
python training/train_fail_risk.py \
  --dataset-glob "logs/*.jsonl" \
  --output models/edon_fail_risk_v1.pt \
  --epochs 100 \
  --batch-size 256 \
  --learning-rate 0.001 \
  --horizon-steps 50
```

**Output:**
- `models/edon_fail_risk_v1.pt` - Trained model
- `models/edon_fail_risk_v1.txt` - Training metrics log

**Sanity Check:**
```bash
python scripts/sanity_check_fail_risk.py \
  --model models/edon_fail_risk_v1.pt \
  --episode logs/edon_train_high_stress_*.jsonl \
  --lookback 30
```

## Step 2: Train v8 Strategy Policy

Train the v8 strategy policy using PPO:

```bash
python training/train_edon_v8_strategy.py \
  --episodes 300 \
  --profile high_stress \
  --seed 0 \
  --lr 5e-4 \
  --gamma 0.995 \
  --update-epochs 10 \
  --output-dir models \
  --model-name edon_v8_strategy_v1 \
  --fail-risk-model models/edon_fail_risk_v1.pt
```

**Output:**
- `models/edon_v8_strategy_v1.pt` - Trained strategy policy

**Training logs show:**
- Per-episode: score, reward, length, interventions, time-to-intervention, near-fail density
- Per-update: policy_loss, entropy, KL divergence
- Every 10 episodes: summary statistics

## Step 3: Evaluate Baseline vs v8

### Option A: Use Helper Script

```bash
python scripts/eval_v8_hell.py
```

This will prompt to run both evaluations automatically, or print commands for manual execution.

### Option B: Manual Execution

**Baseline:**
```bash
python run_eval.py \
  --mode baseline \
  --profile hell_stress \
  --episodes 30 \
  --seed 42 \
  --output results/baseline_v8_hell.json \
  --edon-score
```

**EDON v8:**
```bash
python run_eval.py \
  --mode edon \
  --profile hell_stress \
  --episodes 30 \
  --seed 42 \
  --output results/edon_v8_strategy_hell.json \
  --edon-gain 1.0 \
  --edon-arch v8_strategy \
  --edon-score
```

## Step 4: Compare Results

```bash
python training/compare_v8_vs_baseline.py \
  --baseline results/baseline_v8_hell.json \
  --v8 results/edon_v8_strategy_hell.json
```

**Output includes:**
- Baseline metrics (interventions, stability, time-to-intervention, near-fail density, EDON v8 score)
- v8 metrics (same)
- Deltas (percentage changes)
- Verdict: [PASS] / [NEUTRAL] / [REGRESS]

**Verdict rules:**
- PASS: ΔEDON v8 score ≥ +1.0
- REGRESS: ΔEDON v8 score ≤ -1.0
- NEUTRAL: otherwise

## Expected Output Format

```
================================================================================
EDON v8 vs Baseline Comparison
================================================================================

Baseline:
  Interventions/ep: 45.20
  Stability: 0.0250
  Time-to-first-intervention: 120.5 steps
  Near-fail density: 0.3500
  EDON v8 score: 38.50

EDON v8:
  Interventions/ep: 42.10
  Stability: 0.0220
  Time-to-first-intervention: 145.3 steps
  Near-fail density: 0.2800
  EDON v8 score: 42.30

Deltas:
  ΔInterventions%: -6.9%
  ΔStability%: -12.0%
  ΔTime-to-intervention%: +20.6%
  ΔNear-fail density%: -20.0%
  ΔEDON v8 score: +3.80

Verdict: [PASS]
================================================================================
```

## Files Created/Modified

### Training Scripts
- `training/train_fail_risk.py` - Fail-risk model training
- `training/train_edon_v8_strategy.py` - v8 strategy policy training

### Helper Scripts
- `scripts/sanity_check_fail_risk.py` - Verify fail-risk model
- `scripts/eval_v8_hell.py` - Run baseline vs v8 evaluation

### Comparison
- `training/compare_v8_vs_baseline.py` - Compare results with verdict

### Core Components (already created)
- `training/fail_risk_model.py` - Fail-risk model definition
- `training/edon_v8_policy.py` - Strategy policy definition
- `controllers/edon_v8_reflex.py` - Reflex layer controller
- `env/edon_humanoid_env_v8.py` - v8 environment wrapper
- `metrics/edon_v8_metrics.py` - v8 metrics computation

## Troubleshooting

**Fail-risk model training fails:**
- Check that JSONL files exist and have valid episode data
- Ensure episodes have `type="step"` records with `info` dicts

**v8 training fails:**
- Ensure fail-risk model exists at specified path
- Check that base environment initializes correctly
- Verify input size matches between training and inference

**Evaluation fails:**
- Ensure both models exist (fail-risk and strategy)
- Check that `--edon-arch v8_strategy` is specified
- Verify output directory exists

**Comparison script fails:**
- Check that both JSON result files exist
- Ensure files have `run_metrics` and `episodes` keys
- Missing v8-specific metrics will show as N/A

