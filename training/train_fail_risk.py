"""
Train EDON v8 Fail-Risk Model

Usage:
    python training/train_fail_risk.py \
        --dataset-glob "logs/*.jsonl" \
        --output models/edon_fail_risk_v1.pt \
        --epochs 100 \
        --batch-size 256 \
        --learning-rate 0.001

The fail-risk model predicts probability of failure (intervention/fall) 
in the next 0.5-1.0 seconds, used as a first-class signal in v8 layered control.
"""

import argparse
import glob
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.fail_risk_model import train_fail_risk_model


def main():
    parser = argparse.ArgumentParser(
        description="Train EDON v8 fail-risk model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--dataset-glob",
        type=str,
        required=True,
        help="Glob pattern for JSONL log files (e.g., 'logs/*.jsonl' or 'data/*.jsonl')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/edon_fail_risk_v1.pt",
        help="Output path for trained model"
    )
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=50,
        help="Look-ahead horizon in steps (default: 50 ≈ 0.5-1.0s)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.9,
        help="Train/validation split ratio"
    )
    
    args = parser.parse_args()
    
    # Expand glob pattern
    expanded_paths = glob.glob(args.dataset_glob)
    
    if not expanded_paths:
        print(f"Error: No files found matching pattern: {args.dataset_glob}")
        return
    
    print(f"Found {len(expanded_paths)} dataset file(s)")
    
    # Train model
    train_fail_risk_model(
        dataset_paths=expanded_paths,
        output_path=args.output,
        horizon_steps=args.horizon_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_split=args.train_split
    )


if __name__ == "__main__":
    main()

