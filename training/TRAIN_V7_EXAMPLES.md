# EDON v7 Training Examples

## Training v7 Policy

Train a v7 policy using PPO:

```bash
python training/train_edon_v7.py \
    --episodes 100 \
    --profile high_stress \
    --seed 42 \
    --lr 3e-4 \
    --output-dir models \
    --model-name edon_v7
```

## Evaluating v7 Policy

Evaluate the trained v7 policy:

```bash
python run_eval.py \
    --mode edon \
    --profile high_stress \
    --episodes 30 \
    --seed 42 \
    --output results/edon_v7_highstress.json \
    --edon-gain 1.0 \
    --edon-arch v7_learned \
    --edon-score
```

## Comparing v7 vs Baseline

```bash
# Baseline
python run_eval.py --mode baseline --profile high_stress --episodes 30 --seed 42 --output results/baseline_v7_comparison.json --edon-score

# v7
python run_eval.py --mode edon --profile high_stress --episodes 30 --seed 42 --output results/edon_v7_comparison.json --edon-gain 1.0 --edon-arch v7_learned --edon-score
```

The `--edon-score` flag will print the EDON episode score for each run, allowing you to compare performance.

