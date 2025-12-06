"""
Compare EDON v8 vs Baseline results.

Reads evaluation JSON files and prints comparison metrics with verdict.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.edon_v8_metrics import (
    compute_time_to_first_intervention,
    compute_near_fail_metrics,
    compute_episode_score_v8
)


def load_results(filepath: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {filepath}: {e}")
        sys.exit(1)


def extract_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from results."""
    run_metrics = results.get("run_metrics", {})
    episodes = results.get("episodes", [])
    
    # Basic metrics
    metrics = {
        "interventions": run_metrics.get("interventions_per_episode", 0.0),
        "stability": run_metrics.get("stability_avg", 0.0),
        "episode_length": run_metrics.get("avg_episode_length", 0.0),
        "success_rate": run_metrics.get("success_rate", 0.0)
    }
    
    # Try to extract v8-specific metrics from episodes
    time_to_int_list = []
    near_fail_densities = []
    
    for episode in episodes:
        # Check metadata first
        metadata = episode.get("metadata", {})
        if "time_to_first_intervention" in metadata:
            time_to_int_list.append(metadata["time_to_first_intervention"])
        if "near_fail_density" in metadata:
            near_fail_densities.append(metadata["near_fail_density"])
        
        # Also check episode dict directly
        if "time_to_first_intervention" in episode:
            val = episode["time_to_first_intervention"]
            if val is not None:
                time_to_int_list.append(val)
        if "near_fail_density" in episode:
            near_fail_densities.append(episode["near_fail_density"])
    
    # Compute averages
    if time_to_int_list:
        metrics["time_to_first_intervention"] = sum(time_to_int_list) / len(time_to_int_list)
    else:
        metrics["time_to_first_intervention"] = None
    
    if near_fail_densities:
        metrics["near_fail_density"] = sum(near_fail_densities) / len(near_fail_densities)
    else:
        metrics["near_fail_density"] = 0.0
    
    return metrics


def compute_delta_percent(base: float, new: float) -> float:
    """Compute percentage delta (positive = improvement for most metrics)."""
    if base == 0:
        if new == 0:
            return 0.0
        return float('inf') if new > 0 else float('-inf')
    return 100.0 * (new - base) / abs(base)


def main():
    parser = argparse.ArgumentParser(description="Compare EDON v8 vs Baseline")
    parser.add_argument("--baseline", type=str, required=True, help="Path to baseline results JSON")
    parser.add_argument("--v8", type=str, required=True, help="Path to v8 results JSON")
    
    args = parser.parse_args()
    
    # Load results
    try:
        baseline_results = load_results(args.baseline)
        v8_results = load_results(args.v8)
    except Exception as e:
        print(f"Error loading results: {e}")
        sys.exit(1)
    
    # Extract metrics
    try:
        baseline_metrics = extract_metrics(baseline_results)
        v8_metrics = extract_metrics(v8_results)
    except Exception as e:
        print(f"Error extracting metrics: {e}")
        sys.exit(1)
    
    # Compute EDON scores
    from training.edon_score import compute_episode_score
    
    baseline_summary = {
        "interventions_per_episode": baseline_metrics["interventions"],
        "stability_avg": baseline_metrics["stability"],
        "avg_episode_length": baseline_metrics["episode_length"]
    }
    
    v8_summary = {
        "interventions_per_episode": v8_metrics["interventions"],
        "stability_avg": v8_metrics["stability"],
        "avg_episode_length": v8_metrics["episode_length"],
        "time_to_first_intervention": v8_metrics.get("time_to_first_intervention"),
        "near_fail_density": v8_metrics.get("near_fail_density", 0.0)
    }
    
    try:
        baseline_score = compute_episode_score(baseline_summary)
        v8_score = compute_episode_score_v8(v8_summary)
    except Exception as e:
        print(f"Error computing scores: {e}")
        baseline_score = 0.0
        v8_score = 0.0
    
    # Print comparison
    print("="*80)
    print("EDON v8 vs Baseline Comparison")
    print("="*80)
    print()
    
    # Baseline section
    print("Baseline:")
    print(f"  Interventions/ep: {baseline_metrics['interventions']:.2f}")
    print(f"  Stability: {baseline_metrics['stability']:.4f}")
    time_to_int_base = baseline_metrics.get("time_to_first_intervention")
    if time_to_int_base is not None:
        print(f"  Time-to-first-intervention: {time_to_int_base:.1f} steps")
    else:
        print(f"  Time-to-first-intervention: N/A")
    print(f"  Near-fail density: {baseline_metrics.get('near_fail_density', 0.0):.4f}")
    print(f"  EDON v8 score: {baseline_score:.2f}")
    print()
    
    # v8 section
    print("EDON v8:")
    print(f"  Interventions/ep: {v8_metrics['interventions']:.2f}")
    print(f"  Stability: {v8_metrics['stability']:.4f}")
    time_to_int_v8 = v8_metrics.get("time_to_first_intervention")
    if time_to_int_v8 is not None:
        print(f"  Time-to-first-intervention: {time_to_int_v8:.1f} steps")
    else:
        print(f"  Time-to-first-intervention: N/A")
    print(f"  Near-fail density: {v8_metrics.get('near_fail_density', 0.0):.4f}")
    print(f"  EDON v8 score: {v8_score:.2f}")
    print()
    
    # Deltas section
    print("Deltas:")
    
    # Interventions (lower is better)
    int_delta = compute_delta_percent(baseline_metrics["interventions"], v8_metrics["interventions"])
    print(f"  Delta Interventions%: {int_delta:+.1f}%")
    
    # Stability (lower is better)
    stab_delta = compute_delta_percent(baseline_metrics["stability"], v8_metrics["stability"])
    print(f"  Delta Stability%: {stab_delta:+.1f}%")
    
    # Time-to-intervention (higher is better, so flip sign)
    if time_to_int_base is not None and time_to_int_v8 is not None:
        time_delta = compute_delta_percent(time_to_int_base, time_to_int_v8)
        print(f"  Delta Time-to-intervention%: {time_delta:+.1f}%")
    else:
        print(f"  Delta Time-to-intervention%: N/A")
    
    # Near-fail density (lower is better)
    near_fail_base = baseline_metrics.get("near_fail_density", 0.0)
    near_fail_v8 = v8_metrics.get("near_fail_density", 0.0)
    near_fail_delta = compute_delta_percent(near_fail_base, near_fail_v8)
    print(f"  Delta Near-fail density%: {near_fail_delta:+.1f}%")
    
    # EDON v8 score (higher is better, so flip sign)
    score_delta = v8_score - baseline_score
    print(f"  Delta EDON v8 score: {score_delta:+.2f}")
    print()
    
    # Verdict
    if score_delta >= 1.0:
        verdict = "[PASS]"
    elif score_delta <= -1.0:
        verdict = "[REGRESS]"
    else:
        verdict = "[NEUTRAL]"
    
    print(f"Verdict: {verdict}")
    print("="*80)


if __name__ == "__main__":
    main()

