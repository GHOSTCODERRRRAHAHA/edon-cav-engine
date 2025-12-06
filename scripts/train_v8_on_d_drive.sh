#!/bin/bash
# Train v8 Strategy Policy on D Drive
# This script trains the policy and saves models to D drive

echo "========================================================================"
echo "EDON v8 Strategy Policy Training - D Drive"
echo "========================================================================"
echo ""

# Create D drive directory if it doesn't exist
mkdir -p "D:/edon_models"
mkdir -p "D:/edon_models/logs"

echo "Training will save models to: D:/edon_models/"
echo ""

# Train the policy
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

echo ""
echo "========================================================================"
echo "Training complete!"
echo "Model saved to: D:/edon_models/edon_v8_strategy_v1_trained.pt"
echo "========================================================================"

