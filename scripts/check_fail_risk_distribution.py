"""
Fail-Risk Distribution Check

After retraining with better labeling, verify:
- Positive % target: 40-50%
- Mean fail_risk on safe steps: low
- Mean fail_risk near actual failure: high
If it still looks like "always high," it's not a good control feature.
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.fail_risk_model import FailRiskModel, load_episode_from_jsonl, extract_features_from_step


def analyze_fail_risk_distribution(jsonl_file: str, num_episodes: int = 5):
    """Analyze fail-risk distribution from JSONL logs."""
    print("="*80)
    print("Fail-Risk Distribution Analysis")
    print("="*80)
    
    # Load fail-risk model
    fail_risk_model_path = Path("models/edon_fail_risk_v1_fixed_v2.pt")
    if not fail_risk_model_path.exists():
        print(f"Error: Fail-risk model not found at {fail_risk_model_path}")
        return
    
    checkpoint = torch.load(fail_risk_model_path, map_location="cpu", weights_only=False)
    input_size = checkpoint.get("input_size", 15)
    fail_risk_model = FailRiskModel(input_size=input_size)
    fail_risk_model.load_state_dict(checkpoint["model_state_dict"])
    fail_risk_model.eval()
    
    print(f"Loaded fail-risk model from {fail_risk_model_path}")
    print()
    
    # Load episodes
    episodes = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if data.get("type") == "episode_summary":
                    episodes.append(data)
                elif data.get("type") == "step":
                    # Collect step data
                    pass
    
    # Load step data
    all_steps = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if data.get("type") == "step":
                    all_steps.append(data)
    
    print(f"Loaded {len(episodes)} episodes, {len(all_steps)} steps")
    print()
    
    # Compute fail-risk for all steps
    fail_risks = []
    is_safe = []
    is_failure = []
    
    for step_data in all_steps[:1000]:  # Limit to first 1000 steps for speed
        try:
            features = extract_features_from_step(step_data)
            if features is None:
                continue
            
            with torch.no_grad():
                fail_risk = float(torch.sigmoid(fail_risk_model(torch.FloatTensor(features).unsqueeze(0))).item())
            
            fail_risks.append(fail_risk)
            
            # Check if this is a safe step (no intervention, not near failure)
            interventions = step_data.get("interventions_so_far", 0)
            done = step_data.get("done", False)
            features_dict = step_data.get("features", {})
            tilt_zone = features_dict.get("tilt_zone", "") if isinstance(features_dict, dict) else ""
            
            is_safe_step = (
                interventions == 0 and
                not done and
                tilt_zone not in ["prefall", "fail"]
            )
            is_safe.append(is_safe_step)
            
            # Check if this is a failure step
            is_failure_step = (
                interventions > 0 or
                done or
                tilt_zone in ["fail"]
            )
            is_failure.append(is_failure_step)
            
        except Exception as e:
            continue
    
    if len(fail_risks) == 0:
        print("Error: No valid steps found")
        return
    
    fail_risks = np.array(fail_risks)
    is_safe = np.array(is_safe)
    is_failure = np.array(is_failure)
    
    # Analysis
    print("Fail-Risk Distribution:")
    print(f"  Total steps analyzed: {len(fail_risks)}")
    print(f"  Mean fail-risk: {np.mean(fail_risks):.4f}")
    print(f"  Std fail-risk: {np.std(fail_risks):.4f}")
    print(f"  Min fail-risk: {np.min(fail_risks):.4f}")
    print(f"  Max fail-risk: {np.max(fail_risks):.4f}")
    print()
    
    # Safe vs failure
    safe_fail_risks = fail_risks[is_safe]
    failure_fail_risks = fail_risks[is_failure]
    
    print("Safe Steps (no intervention, not near failure):")
    print(f"  Count: {len(safe_fail_risks)}")
    if len(safe_fail_risks) > 0:
        print(f"  Mean fail-risk: {np.mean(safe_fail_risks):.4f}")
        print(f"  Std fail-risk: {np.std(safe_fail_risks):.4f}")
    print()
    
    print("Failure Steps (intervention or actual failure):")
    print(f"  Count: {len(failure_fail_risks)}")
    if len(failure_fail_risks) > 0:
        print(f"  Mean fail-risk: {np.mean(failure_fail_risks):.4f}")
        print(f"  Std fail-risk: {np.std(failure_fail_risks):.4f}")
    print()
    
    # Check if fail-risk discriminates
    if len(safe_fail_risks) > 0 and len(failure_fail_risks) > 0:
        mean_safe = np.mean(safe_fail_risks)
        mean_failure = np.mean(failure_fail_risks)
        separation = mean_failure - mean_safe
        
        print("Discrimination Analysis:")
        print(f"  Mean fail-risk (safe): {mean_safe:.4f}")
        print(f"  Mean fail-risk (failure): {mean_failure:.4f}")
        print(f"  Separation: {separation:.4f}")
        print()
        
        if separation < 0.1:
            print("[WARNING] Fail-risk does NOT discriminate well!")
            print("   Mean fail-risk is similar for safe and failure steps.")
            print("   This is not a useful control feature.")
        elif separation < 0.2:
            print("[CAUTION] Fail-risk discrimination is weak.")
            print("   Consider improving the model or using different features.")
        else:
            print("[GOOD] Fail-risk discriminates well between safe and failure steps.")
    
    # Check positive percentage (fail-risk > 0.5)
    positive_count = np.sum(fail_risks > 0.5)
    positive_pct = 100.0 * positive_count / len(fail_risks)
    
    print(f"Positive Rate (fail-risk > 0.5): {positive_pct:.1f}%")
    if positive_pct < 30 or positive_pct > 70:
        print("[WARNING] Positive rate outside ideal range (40-50%)")
    else:
        print("[OK] Positive rate is in reasonable range")
    
    print("="*80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, default="logs/edon_train_high_stress_20251126_033041.jsonl", help="JSONL log file")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to analyze")
    args = parser.parse_args()
    
    analyze_fail_risk_distribution(args.jsonl, args.episodes)

