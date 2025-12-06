"""
Sanity check script for fail-risk model.

Loads a trained fail-risk model and recent JSONL episodes,
then prints a table showing fail_risk predictions before failures.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from training.fail_risk_model import FailRiskModel, extract_features_from_step, load_episode_from_jsonl


def main():
    parser = argparse.ArgumentParser(description="Sanity check fail-risk model")
    parser.add_argument(
        "--model",
        type=str,
        default="models/edon_fail_risk_v1.pt",
        help="Path to trained fail-risk model"
    )
    parser.add_argument(
        "--episode",
        type=str,
        required=True,
        help="Path to JSONL episode file"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=30,
        help="Number of steps to show before failure (default: 30)"
    )
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {args.model}")
        return
    
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    input_size = checkpoint.get("input_size", 15)
    model = FailRiskModel(input_size=input_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Model loaded: input_size={input_size}")
    
    # Load episode
    print(f"\nLoading episode from {args.episode}...")
    episode_path = Path(args.episode)
    if not episode_path.exists():
        print(f"Error: Episode file not found at {args.episode}")
        return
    
    episodes = load_episode_from_jsonl(episode_path)
    if len(episodes) == 0:
        print("Error: No episodes found in file")
        return
    
    # Use first episode
    episode = episodes[0]
    print(f"  Episode loaded: {len(episode)} steps")
    
    # Find first intervention/fall
    first_failure_idx = None
    for i, step in enumerate(episode):
        info = step.get("info", {})
        if isinstance(info, dict):
            intervention = info.get("intervention", False)
            fallen = info.get("fallen", False)
        else:
            intervention = step.get("intervention", False)
            fallen = step.get("fallen", False)
        
        if intervention or fallen:
            first_failure_idx = i
            break
    
    if first_failure_idx is None:
        print("  No intervention/fall found in episode")
        # Show last N steps anyway
        first_failure_idx = len(episode)
    
    # Compute fail_risk for each step
    print(f"\nComputing fail_risk for steps...")
    results = []
    
    for i, step_data in enumerate(episode):
        try:
            features = extract_features_from_step(step_data)
            
            # Run inference
            with torch.no_grad():
                feature_tensor = torch.FloatTensor(features).unsqueeze(0)
                fail_risk = model(feature_tensor).item()
            
            # Extract state
            obs = step_data.get("obs") or step_data.get("state") or step_data
            roll = float(obs.get("roll", 0.0)) if isinstance(obs, dict) else 0.0
            pitch = float(obs.get("pitch", 0.0)) if isinstance(obs, dict) else 0.0
            tilt_mag = np.sqrt(roll**2 + pitch**2)
            
            # Check for intervention/fall
            info = step_data.get("info", {})
            if isinstance(info, dict):
                intervention = info.get("intervention", False)
                fallen = info.get("fallen", False)
            else:
                intervention = step_data.get("intervention", False)
                fallen = step_data.get("fallen", False)
            
            results.append({
                "t": i,
                "tilt_mag": tilt_mag,
                "intervention": intervention,
                "fallen": fallen,
                "fail_risk": fail_risk
            })
        except Exception as e:
            continue
    
    # Show table for steps before failure
    start_idx = max(0, first_failure_idx - args.lookback)
    end_idx = min(len(results), first_failure_idx + 1)
    
    print(f"\n{'='*80}")
    print(f"Fail-Risk Analysis (steps {start_idx} to {end_idx-1}, failure at step {first_failure_idx})")
    print(f"{'='*80}")
    print(f"{'t':<6} {'tilt_mag':<10} {'intervention':<12} {'fallen':<8} {'fail_risk':<10}")
    print("-"*80)
    
    for r in results[start_idx:end_idx]:
        intervention_str = "YES" if r["intervention"] else "no"
        fallen_str = "YES" if r["fallen"] else "no"
        fail_risk_str = f"{r['fail_risk']:.4f}"
        
        # Highlight high fail_risk
        if r["fail_risk"] > 0.6:
            fail_risk_str = f"***{fail_risk_str}***"
        
        print(f"{r['t']:<6} {r['tilt_mag']:<10.4f} {intervention_str:<12} {fallen_str:<8} {fail_risk_str:<10}")
    
    print(f"{'='*80}")
    
    # Summary stats
    if first_failure_idx > 0:
        pre_failure_risks = [r["fail_risk"] for r in results[:first_failure_idx]]
        if pre_failure_risks:
            avg_pre_failure = np.mean(pre_failure_risks)
            max_pre_failure = np.max(pre_failure_risks)
            print(f"\nPre-failure statistics:")
            print(f"  Average fail_risk: {avg_pre_failure:.4f}")
            print(f"  Max fail_risk: {max_pre_failure:.4f}")
            print(f"  Steps with fail_risk > 0.6: {sum(1 for r in pre_failure_risks if r > 0.6)}")


if __name__ == "__main__":
    main()

