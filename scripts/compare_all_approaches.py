"""
Compare All Approaches: Baseline vs v8 Retrained vs v7-Style

Creates a comprehensive comparison table.
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.compare_v8_vs_baseline import load_results, extract_metrics
from metrics.edon_v8_metrics import compute_episode_score_v8


def compare_all(baseline_file: str, v8_file: str, v7_file: str):
    """Compare all three approaches."""
    print("="*80)
    print("Comprehensive Comparison: Baseline vs v8 Retrained vs v7-Style")
    print("="*80)
    print()
    
    # Load all results
    baseline_data = load_results(baseline_file)
    v8_data = load_results(v8_file)
    v7_data = load_results(v7_file) if Path(v7_file).exists() else None
    
    # Extract metrics
    baseline_metrics = extract_metrics(baseline_data)
    v8_metrics = extract_metrics(v8_data)
    v7_metrics = extract_metrics(v7_data) if v7_data else None
    
    # Compute EDON scores
    baseline_run_metrics = baseline_data.get("run_metrics", {})
    baseline_summary = {
        "interventions_per_episode": baseline_run_metrics.get("interventions_per_episode", 0),
        "stability_avg": baseline_run_metrics.get("stability_avg", 0.0),
        "avg_episode_length": baseline_run_metrics.get("avg_episode_length", 0)
    }
    baseline_metrics["edon_score_v8"] = compute_episode_score_v8(baseline_summary)
    
    v8_run_metrics = v8_data.get("run_metrics", {})
    v8_summary = {
        "interventions_per_episode": v8_run_metrics.get("interventions_per_episode", 0),
        "stability_avg": v8_run_metrics.get("stability_avg", 0.0),
        "avg_episode_length": v8_run_metrics.get("avg_episode_length", 0)
    }
    v8_metrics["edon_score_v8"] = compute_episode_score_v8(v8_summary)
    
    if v7_data:
        v7_run_metrics = v7_data.get("run_metrics", {})
        v7_summary = {
            "interventions_per_episode": v7_run_metrics.get("interventions_per_episode", 0),
            "stability_avg": v7_run_metrics.get("stability_avg", 0.0),
            "avg_episode_length": v7_run_metrics.get("avg_episode_length", 0)
        }
        v7_metrics["edon_score_v8"] = compute_episode_score_v8(v7_summary)
    
    # Print comparison table
    print(f"{'Metric':<30} {'Baseline':<15} {'v8 Retrained':<15} {'v7-Style':<15}")
    print("-" * 80)
    
    metrics_to_compare = [
        ("Interventions/ep", "interventions"),
        ("Stability", "stability"),
        ("Time-to-intervention", "time_to_first_intervention"),
        ("Near-fail density", "near_fail_density"),
        ("EDON v8 score", "edon_score_v8")
    ]
    
    for metric_name, metric_key in metrics_to_compare:
        baseline_val = baseline_metrics.get(metric_key, "N/A")
        v8_val = v8_metrics.get(metric_key, "N/A")
        v7_val = v7_metrics.get(metric_key, "N/A") if v7_metrics else "N/A"
        
        # Format values
        if isinstance(baseline_val, float):
            baseline_str = f"{baseline_val:.2f}"
        else:
            baseline_str = str(baseline_val)
        
        if isinstance(v8_val, float):
            v8_str = f"{v8_val:.2f}"
        else:
            v8_str = str(v8_val)
        
        if isinstance(v7_val, float):
            v7_str = f"{v7_val:.2f}"
        else:
            v7_str = str(v7_val)
        
        print(f"{metric_name:<30} {baseline_str:<15} {v8_str:<15} {v7_str:<15}")
    
    print()
    print("="*80)
    
    # Compute deltas
    print("\nDeltas vs Baseline:")
    print("-" * 80)
    
    for metric_name, metric_key in metrics_to_compare:
        baseline_val = baseline_metrics.get(metric_key)
        v8_val = v8_metrics.get(metric_key)
        v7_val = v7_metrics.get(metric_key) if v7_metrics else None
        
        if baseline_val is None or not isinstance(baseline_val, (int, float)):
            continue
        
        # v8 delta
        if v8_val is not None and isinstance(v8_val, (int, float)):
            v8_delta = ((v8_val - baseline_val) / baseline_val * 100) if baseline_val != 0 else 0
            v8_delta_str = f"{v8_delta:+.1f}%"
        else:
            v8_delta_str = "N/A"
        
        # v7 delta
        if v7_val is not None and isinstance(v7_val, (int, float)):
            v7_delta = ((v7_val - baseline_val) / baseline_val * 100) if baseline_val != 0 else 0
            v7_delta_str = f"{v7_delta:+.1f}%"
        else:
            v7_delta_str = "N/A"
        
        print(f"{metric_name:<30} v8: {v8_delta_str:<10} v7: {v7_delta_str:<10}")
    
    print()
    print("="*80)
    
    # Verdicts
    print("\nVerdicts:")
    print("-" * 80)
    
    baseline_score = baseline_metrics.get("edon_score_v8", 0)
    v8_score = v8_metrics.get("edon_score_v8", 0)
    v7_score = v7_metrics.get("edon_score_v8", 0) if v7_metrics else None
    
    if isinstance(baseline_score, (int, float)) and isinstance(v8_score, (int, float)):
        v8_delta = v8_score - baseline_score
        if v8_delta >= 1.0:
            v8_verdict = "[PASS]"
        elif v8_delta <= -1.0:
            v8_verdict = "[REGRESS]"
        else:
            v8_verdict = "[NEUTRAL]"
        print(f"v8 Retrained: {v8_verdict} (Delta: {v8_delta:+.2f})")
    
    if isinstance(baseline_score, (int, float)) and v7_score is not None and isinstance(v7_score, (int, float)):
        v7_delta = v7_score - baseline_score
        if v7_delta >= 1.0:
            v7_verdict = "[PASS]"
        elif v7_delta <= -1.0:
            v7_verdict = "[REGRESS]"
        else:
            v7_verdict = "[NEUTRAL]"
        print(f"v7-Style: {v7_verdict} (Delta: {v7_delta:+.2f})")
    
    print("="*80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default="results/baseline_v8_final.json")
    parser.add_argument("--v8", type=str, default="results/v8_retrained_no_reflex.json")
    parser.add_argument("--v7", type=str, default="results/v7_improved.json")
    args = parser.parse_args()
    
    compare_all(args.baseline, args.v8, args.v7)

