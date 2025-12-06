# EDON v8 Architecture

EDON v8 is a layered adaptive controller that outputs strategies and modulation signals (not raw action deltas), with a predictive failure model and bad-condition metrics.

## Architecture Overview

v8 uses a two-layer control architecture:

1. **Reflex Layer** (Deterministic): Fast micro-stabilization with damping and guardrails
2. **Strategy Layer** (Learned): Slow RL/learned policy that outputs strategy selection and modulation signals

The strategy layer outputs:
- **strategy_id**: Discrete strategy selection (NORMAL, HIGH_DAMPING, RECOVERY_BALANCE, COMPLIANT_TERRAIN)
- **modulations**: Continuous signals (gain_scale, lateral_compliance, step_height_bias)

The reflex layer applies deterministic adjustments based on:
- Current tilt and velocity
- Fail-risk prediction
- Strategy modulations

## Components

### 1. Predictive Failure Model

The fail-risk model predicts probability of failure (intervention/fall) in the next 0.5-1.0 seconds.

**Train fail-risk model:**
```bash
python training/train_fail_risk.py \
  --dataset-paths logs/*.jsonl \
  --output models/edon_fail_risk_v1.pt \
  --horizon-steps 50 \
  --epochs 100 \
  --batch-size 256 \
  --learning-rate 0.001
```

**Files:**
- `training/fail_risk_model.py`: Model definition and training utilities
- `training/train_fail_risk.py`: CLI training script

### 2. Reflex Layer

Deterministic controller that applies smooth damping and guardrails.

**Files:**
- `controllers/edon_v8_reflex.py`: Reflex controller implementation

### 3. Strategy Layer

Learned policy that outputs strategy and modulations.

**Files:**
- `training/edon_v8_policy.py`: Strategy policy network definition

### 4. v8 Environment Wrapper

Environment wrapper that applies layered control.

**Files:**
- `env/edon_humanoid_env_v8.py`: v8 environment wrapper

### 5. v8 Metrics

Metrics specific to v8 including time-to-intervention, fail-risk, and near-fail density.

**Files:**
- `metrics/edon_v8_metrics.py`: v8 metrics computation

## Training

### Step 1: Train Fail-Risk Model

First, train the fail-risk model from existing JSONL logs:

```bash
python training/train_fail_risk.py \
  --dataset-paths logs/edon_train_*.jsonl \
  --output models/edon_fail_risk_v1.pt \
  --horizon-steps 50 \
  --epochs 100
```

### Step 2: Train v8 Strategy Policy

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

**Key parameters:**
- `--episodes`: Number of training episodes (default: 300)
- `--profile`: Stress profile (high_stress, hell_stress, etc.)
- `--lr`: Learning rate (default: 5e-4)
- `--gamma`: Discount factor (default: 0.995)
- `--update-epochs`: PPO update epochs (default: 10)
- `--fail-risk-model`: Path to trained fail-risk model

## Evaluation

### Baseline Evaluation

```bash
python run_eval.py \
  --mode baseline \
  --profile hell_stress \
  --episodes 30 \
  --seed 42 \
  --output results/baseline_v8_hell.json \
  --edon-score
```

### EDON v8 Evaluation

**Note:** v8 requires special evaluation handling. The v8 environment wrapper must be used directly in the evaluation loop. For now, v8 evaluation can be done by:

1. Using the v8 training script in evaluation mode (modify `train_edon_v8_strategy.py`)
2. Or integrating v8 environment wrapper into `run_eval.py` evaluation loop

**Planned integration:**
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

## Comparison

Compare v8 vs baseline results:

```bash
python training/compare_v8_vs_baseline.py \
  --baseline results/baseline_v8_hell.json \
  --v8 results/edon_v8_strategy_hell.json
```

**Output includes:**
- Interventions/episode
- Stability (avg)
- Episode length (avg)
- Success rate
- Time to first intervention (v8 only)
- Average fail-risk (v8 only)
- Near-fail density (v8 only)
- EDON Score v8

## Key Differences from v7

1. **No raw action deltas**: v8 outputs strategies and modulations, not 10-dim action deltas
2. **Layered control**: Reflex layer + Strategy layer
3. **Fail-risk integration**: First-class fail-risk signal used throughout
4. **v8 metrics**: Time-to-intervention, near-fail density, etc.
5. **Environment wrapper**: v8 uses special environment wrapper that handles layered control

## Files Structure

```
training/
  fail_risk_model.py          # Fail-risk model definition
  train_fail_risk.py          # Fail-risk training script
  edon_v8_policy.py           # Strategy policy definition
  train_edon_v8_strategy.py   # v8 strategy training script
  compare_v8_vs_baseline.py   # Comparison script
  V8_README.md               # This file

controllers/
  edon_v8_reflex.py          # Reflex layer controller

env/
  edon_humanoid_env_v8.py    # v8 environment wrapper

metrics/
  edon_v8_metrics.py         # v8 metrics computation
```

## Troubleshooting

**Fail-risk model not loading:**
- Ensure model file exists at `models/edon_fail_risk_v1.pt`
- Check that model was trained with compatible feature extraction

**v8 environment wrapper errors:**
- Ensure fail-risk model is loaded correctly
- Check that strategy policy model exists and is compatible

**Training not converging:**
- Adjust learning rate (try 1e-4 or 5e-5)
- Increase update epochs (try 12 or 15)
- Check reward scaling in environment

## Future Improvements

- [ ] Full integration into `run_eval.py` evaluation loop
- [ ] Support for multiple strategy types
- [ ] Adaptive horizon for fail-risk prediction
- [ ] Online fail-risk model updates
- [ ] Multi-objective optimization for strategies

