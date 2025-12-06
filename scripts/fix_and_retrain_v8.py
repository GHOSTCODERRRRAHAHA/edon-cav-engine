"""
Fix and Retrain v8 Pipeline

This script:
1. Retrains fail-risk model with improved label detection
2. Trains v8 strategy policy for proper number of episodes
"""

import subprocess
import sys
from pathlib import Path


def main():
    print("="*80)
    print("EDON v8 Fix and Retrain Pipeline")
    print("="*80)
    print()
    
    # Step 1: Retrain fail-risk model with fixed label detection
    print("[1/2] Retraining fail-risk model with improved label detection...")
    print("-"*80)
    
    cmd1 = [
        "python", "training/train_fail_risk.py",
        "--dataset-glob", "logs/*.jsonl",
        "--output", "models/edon_fail_risk_v1_fixed.pt",
        "--epochs", "100",
        "--batch-size", "256",
        "--learning-rate", "0.001",
        "--horizon-steps", "50"
    ]
    
    result1 = subprocess.run(cmd1)
    if result1.returncode != 0:
        print("ERROR: Fail-risk model training failed!")
        return
    
    print("\n[2/2] Training v8 strategy policy (300 episodes)...")
    print("-"*80)
    print("NOTE: This will take a while. Training for 300 episodes...")
    print()
    
    cmd2 = [
        "python", "training/train_edon_v8_strategy.py",
        "--episodes", "300",
        "--profile", "high_stress",
        "--seed", "0",
        "--lr", "5e-4",
        "--gamma", "0.995",
        "--update-epochs", "10",
        "--output-dir", "models",
        "--model-name", "edon_v8_strategy_v1_trained",
        "--fail-risk-model", "models/edon_fail_risk_v1_fixed.pt",
        "--max-steps", "1000"
    ]
    
    print("Command:")
    print("  " + " ".join(cmd2))
    print()
    
    response = input("Start training? (y/n): ").strip().lower()
    if response == 'y':
        result2 = subprocess.run(cmd2)
        if result2.returncode == 0:
            print("\n" + "="*80)
            print("Training complete!")
            print("="*80)
            print("\nModels saved:")
            print("  - models/edon_fail_risk_v1_fixed.pt")
            print("  - models/edon_v8_strategy_v1_trained.pt")
            print("\nNext steps:")
            print("  1. Evaluate baseline: python run_eval.py --mode baseline --profile high_stress --episodes 30 --seed 42 --output results/baseline_v8_final.json --edon-score")
            print("  2. Evaluate v8: python run_eval.py --mode edon --profile high_stress --episodes 30 --seed 42 --output results/edon_v8_final.json --edon-gain 1.0 --edon-arch v8_strategy --edon-score")
            print("  3. Compare: python training/compare_v8_vs_baseline.py --baseline results/baseline_v8_final.json --v8 results/edon_v8_final.json")
        else:
            print("\nERROR: v8 strategy training failed!")
    else:
        print("\nTraining cancelled. Run the command manually when ready.")


if __name__ == "__main__":
    main()

