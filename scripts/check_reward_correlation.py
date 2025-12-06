"""
Reward Sanity Check

Logs average per-episode reward vs EDON v8 score and computes correlation.
If correlation < 0.5, PPO is not optimizing what we care about.
"""

import sys
from pathlib import Path
import json
import numpy as np
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.edon_v8_metrics import compute_episode_score_v8


def analyze_reward_correlation(result_file: str):
    """Analyze correlation between reward and EDON v8 score."""
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    episodes = data.get("episodes", [])
    
    rewards = []
    edon_scores = []
    
    for episode in episodes:
        # Extract reward (if available in metadata or computed)
        # For now, we'll need to compute it from episode data
        # Or check if it's already in the episode dict
        
        # Compute EDON v8 score
        episode_summary = {
            "interventions_per_episode": episode.get("interventions", 0),
            "stability_avg": episode.get("stability_score", 0.0),
            "avg_episode_length": episode.get("episode_length", 0),
            "time_to_first_intervention": episode.get("metadata", {}).get("time_to_first_intervention"),
            "near_fail_density": episode.get("metadata", {}).get("near_fail_density", 0.0)
        }
        
        edon_score = compute_episode_score_v8(episode_summary)
        edon_scores.append(edon_score)
        
        # Try to get reward from metadata
        metadata = episode.get("metadata", {})
        if "total_reward" in metadata:
            rewards.append(metadata["total_reward"])
        elif "avg_reward" in metadata:
            rewards.append(metadata["avg_reward"] * episode.get("episode_length", 0))
        else:
            # Estimate from episode length and interventions (rough approximation)
            # This is not ideal, but we need some proxy
            length = episode.get("episode_length", 0)
            interventions = episode.get("interventions", 0)
            # Rough estimate: -1.5 per step, -25 per intervention, +0.8 per step alive
            estimated_reward = length * (-1.5 + 0.8) - interventions * 25.0
            rewards.append(estimated_reward)
    
    if len(rewards) < 2:
        print("Not enough data for correlation analysis")
        return
    
    # Compute correlation
    correlation = np.corrcoef(rewards, edon_scores)[0, 1]
    
    print("="*80)
    print("Reward vs EDON v8 Score Correlation Analysis")
    print("="*80)
    print(f"Episodes analyzed: {len(rewards)}")
    print(f"Reward range: [{min(rewards):.1f}, {max(rewards):.1f}]")
    print(f"EDON v8 Score range: [{min(edon_scores):.1f}, {max(edon_scores):.1f}]")
    print()
    print(f"Correlation: {correlation:.3f}")
    print()
    
    if correlation < 0.5:
        print("[WARNING] Correlation < 0.5")
        print("   PPO is NOT optimizing what you care about!")
        print("   The reward function is misaligned with EDON v8 score.")
    elif correlation < 0.7:
        print("[CAUTION] Correlation < 0.7")
        print("   Reward function could be better aligned.")
    else:
        print("[GOOD] Correlation >= 0.7")
        print("   Reward function is well-aligned with EDON v8 score.")
    
    print("="*80)
    
    return correlation


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=str, default="results/edon_v8_fixed_final.json", help="Evaluation result file")
    args = parser.parse_args()
    
    analyze_reward_correlation(args.result)

