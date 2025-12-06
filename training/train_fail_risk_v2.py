"""
Train Fail-Risk Model v2

Uses improved features (temporal + energy) with XGBoost/LightGBM.
Target: fail-risk separation >= 0.25
"""

import argparse
import sys
from pathlib import Path
import glob

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.fail_risk_model_v2 import train_fail_risk_model_v2


def main():
    parser = argparse.ArgumentParser(description="Train fail-risk model v2")
    parser.add_argument("--dataset-glob", type=str, required=True, help="Glob pattern for JSONL dataset files")
    parser.add_argument("--output", type=str, default="models/edon_fail_risk_v2.pt", help="Output model path")
    parser.add_argument("--horizon-steps", type=int, default=50, help="Horizon for failure prediction")
    parser.add_argument("--model-type", type=str, default="lightgbm", choices=["lightgbm", "xgboost"], help="Model type")
    
    args = parser.parse_args()
    
    # Expand glob pattern
    dataset_paths = glob.glob(args.dataset_glob)
    if not dataset_paths:
        print(f"[ERROR] No files found matching pattern: {args.dataset_glob}")
        return
    
    print(f"Found {len(dataset_paths)} dataset file(s)")
    
    # Train model
    results = train_fail_risk_model_v2(
        dataset_paths=dataset_paths,
        output_path=args.output,
        horizon_steps=args.horizon_steps,
        model_type=args.model_type
    )
    
    if results:
        print()
        print("="*80)
        print("Training Complete!")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  AUC: {results['auc']:.4f}")
        print(f"  Separation: {results['separation']:.4f}")
        if results['separation'] >= 0.25:
            print("  [SUCCESS] Separation >= 0.25 target achieved!")
        else:
            print(f"  [WARNING] Separation {results['separation']:.4f} < 0.25 target")
        print("="*80)


if __name__ == "__main__":
    main()

