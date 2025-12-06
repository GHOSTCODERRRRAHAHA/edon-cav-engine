#!/usr/bin/env python3
"""
Plot comparison between baseline and EDON evaluation results.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional


def load_results(path: Path) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def compute_improvements(baseline: Dict[str, Any], edon: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute percentage improvements.
    
    Returns:
        Dictionary with improvement percentages
    """
    improvements = {}
    
    # Interventions (lower is better)
    baseline_interventions = baseline["interventions_per_episode"]
    edon_interventions = edon["interventions_per_episode"]
    if baseline_interventions > 0:
        improvements["intervention_reduction_pct"] = (
            (baseline_interventions - edon_interventions) / baseline_interventions * 100
        )
    else:
        improvements["intervention_reduction_pct"] = 0.0
    
    # Prefall events (lower is better - EDON should reduce time in prefall zone)
    baseline_prefall = baseline.get("prefall_events_per_episode", 0.0)
    edon_prefall = edon.get("prefall_events_per_episode", 0.0)
    if baseline_prefall > 0:
        improvements["prefall_reduction_pct"] = (
            (baseline_prefall - edon_prefall) / baseline_prefall * 100
        )
    else:
        improvements["prefall_reduction_pct"] = 0.0
    
    # Prefall time (lower is better)
    baseline_prefall_time = baseline.get("prefall_time_avg", 0.0)
    edon_prefall_time = edon.get("prefall_time_avg", 0.0)
    if baseline_prefall_time > 0:
        improvements["prefall_time_reduction_pct"] = (
            (baseline_prefall_time - edon_prefall_time) / baseline_prefall_time * 100
        )
    else:
        improvements["prefall_time_reduction_pct"] = 0.0
    
    # Freeze events (lower is better)
    baseline_freezes = baseline["freeze_events_per_episode"]
    edon_freezes = edon["freeze_events_per_episode"]
    if baseline_freezes > 0:
        improvements["freeze_reduction_pct"] = (
            (baseline_freezes - edon_freezes) / baseline_freezes * 100
        )
    else:
        improvements["freeze_reduction_pct"] = 0.0
    
    # Stability (lower is better)
    baseline_stability = baseline["stability_avg"]
    edon_stability = edon["stability_avg"]
    if baseline_stability > 0:
        improvements["stability_improvement_pct"] = (
            (baseline_stability - edon_stability) / baseline_stability * 100
        )
    else:
        improvements["stability_improvement_pct"] = 0.0
    
    # Success rate (higher is better)
    baseline_success = baseline["success_rate"]
    edon_success = edon["success_rate"]
    if baseline_success > 0:
        improvements["success_improvement_pct"] = (
            (edon_success - baseline_success) / baseline_success * 100
        )
    else:
        improvements["success_improvement_pct"] = (
            edon_success * 100 if edon_success > 0 else 0.0
        )
    
    return improvements


def plot_comparison(
    baseline: Dict[str, Any],
    edon: Dict[str, Any],
    output_dir: Path
) -> None:
    """Generate comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute improvements
    improvements = compute_improvements(baseline, edon)
    
    # Print summary
    print("=" * 70)
    print("EDON vs BASELINE COMPARISON")
    print("=" * 70)
    print(f"Intervention reduction: {improvements['intervention_reduction_pct']:.1f}%")
    print(f"Prefall events reduction: {improvements.get('prefall_reduction_pct', 0.0):.1f}%")
    print(f"Prefall time reduction: {improvements.get('prefall_time_reduction_pct', 0.0):.1f}%")
    print(f"Freeze reduction: {improvements['freeze_reduction_pct']:.1f}%")
    print(f"Stability improvement: {improvements['stability_improvement_pct']:.1f}%")
    print(f"Success rate improvement: {improvements['success_improvement_pct']:.1f}%")
    print("=" * 70)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('EDON vs Baseline Comparison', fontsize=16, fontweight='bold')
    
    # 1. Interventions per episode
    ax = axes[0, 0]
    categories = ['Baseline', 'EDON']
    values = [
        baseline["interventions_per_episode"],
        edon["interventions_per_episode"]
    ]
    colors = ['#e74c3c', '#2ecc71']
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Interventions per Episode', fontweight='bold')
    ax.set_title('Interventions (Lower is Better)')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Add improvement annotation
    if improvements['intervention_reduction_pct'] > 0:
        ax.text(0.5, max(values) * 0.8,
                f'↓ {improvements["intervention_reduction_pct"]:.1f}%',
                ha='center', fontsize=10, color='green', fontweight='bold')
    
    # 2. Freeze events per episode
    ax = axes[0, 1]
    values = [
        baseline["freeze_events_per_episode"],
        edon["freeze_events_per_episode"]
    ]
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Freeze Events per Episode', fontweight='bold')
    ax.set_title('Freeze Events (Lower is Better)')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    if improvements['freeze_reduction_pct'] > 0:
        ax.text(0.5, max(values) * 0.8,
                f'↓ {improvements["freeze_reduction_pct"]:.1f}%',
                ha='center', fontsize=10, color='green', fontweight='bold')
    
    # 3. Stability score
    ax = axes[1, 0]
    values = [
        baseline["stability_avg"],
        edon["stability_avg"]
    ]
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Stability Score', fontweight='bold')
    ax.set_title('Stability (Lower is Better)')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    if improvements['stability_improvement_pct'] > 0:
        ax.text(0.5, max(values) * 0.8,
                f'↓ {improvements["stability_improvement_pct"]:.1f}%',
                ha='center', fontsize=10, color='green', fontweight='bold')
    
    # 4. Success rate
    ax = axes[1, 1]
    values = [
        baseline["success_rate"],
        edon["success_rate"]
    ]
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Success Rate', fontweight='bold')
    ax.set_title('Success Rate (Higher is Better)')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2%}',
                ha='center', va='bottom', fontweight='bold')
    
    if improvements['success_improvement_pct'] > 0:
        ax.text(0.5, 0.8,
                f'↑ {improvements["success_improvement_pct"]:.1f}%',
                ha='center', fontsize=10, color='green', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to: {plot_path}")
    
    # Plot time-series stability if available
    plot_stability_timeseries(baseline, edon, output_dir)
    
    plt.close()


def plot_stability_timeseries(
    baseline: Dict[str, Any],
    edon: Dict[str, Any],
    output_dir: Path
) -> None:
    """Plot stability time-series for sample episodes."""
    # Find episodes with roll/pitch history
    baseline_episodes = baseline.get("raw_episode_metrics", [])
    edon_episodes = edon.get("raw_episode_metrics", [])
    
    if not baseline_episodes or not edon_episodes:
        print("No per-episode data available for time-series plot")
        return
    
    # Find a representative episode (middle one)
    baseline_idx = len(baseline_episodes) // 2
    edon_idx = len(edon_episodes) // 2
    
    baseline_ep = baseline_episodes[baseline_idx]
    edon_ep = edon_episodes[edon_idx]
    
    # Check if we have roll/pitch history
    if "roll_history" not in baseline_ep or "pitch_history" not in baseline_ep:
        print("No roll/pitch history available for time-series plot")
        return
    
    # Create time-series plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Stability Time-Series Comparison (Sample Episodes)', fontsize=14, fontweight='bold')
    
    # Roll
    ax = axes[0]
    baseline_roll = baseline_ep.get("roll_history", [])
    edon_roll = edon_ep.get("roll_history", [])
    
    if baseline_roll and edon_roll:
        steps = range(len(baseline_roll))
        ax.plot(steps, baseline_roll, label='Baseline', color='#e74c3c', alpha=0.7, linewidth=1.5)
        ax.plot(steps[:len(edon_roll)], edon_roll, label='EDON', color='#2ecc71', alpha=0.7, linewidth=1.5)
        ax.set_ylabel('Roll (radians)', fontweight='bold')
        ax.set_title('Torso Roll Over Time')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Pitch
    ax = axes[1]
    baseline_pitch = baseline_ep.get("pitch_history", [])
    edon_pitch = edon_ep.get("pitch_history", [])
    
    if baseline_pitch and edon_pitch:
        steps = range(len(baseline_pitch))
        ax.plot(steps, baseline_pitch, label='Baseline', color='#e74c3c', alpha=0.7, linewidth=1.5)
        ax.plot(steps[:len(edon_pitch)], edon_pitch, label='EDON', color='#2ecc71', alpha=0.7, linewidth=1.5)
        ax.set_xlabel('Step', fontweight='bold')
        ax.set_ylabel('Pitch (radians)', fontweight='bold')
        ax.set_title('Torso Pitch Over Time')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "stability_timeseries.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved time-series plot to: {plot_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot comparison between baseline and EDON results"
    )
    
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline results JSON"
    )
    
    parser.add_argument(
        "--edon",
        type=str,
        required=True,
        help="Path to EDON results JSON"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="plots",
        help="Output directory for plots (default: plots/)"
    )
    
    args = parser.parse_args()
    
    # Load results
    baseline_path = Path(args.baseline)
    edon_path = Path(args.edon)
    
    if not baseline_path.exists():
        print(f"ERROR: Baseline results not found: {baseline_path}")
        sys.exit(1)
    
    if not edon_path.exists():
        print(f"ERROR: EDON results not found: {edon_path}")
        sys.exit(1)
    
    print(f"Loading baseline results from: {baseline_path}")
    baseline = load_results(baseline_path)
    
    print(f"Loading EDON results from: {edon_path}")
    edon = load_results(edon_path)
    
    # Generate plots
    output_dir = Path(args.output)
    plot_comparison(baseline, edon, output_dir)
    
    print("\nPlotting complete!")


if __name__ == "__main__":
    import sys
    main()

