#!/usr/bin/env python3
"""
EDON v6 Validation Script

Runs baseline and v6_learned evaluations and compares results.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple


def extract_metrics(json_path: str) -> Tuple[float, float, float]:
    """
    Extract metrics from JSON results file.
    
    Returns:
        (interventions_per_episode, stability_avg, avg_episode_length)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Handle both old and new JSON formats
    if "run_metrics" in data:
        metrics = data["run_metrics"]
    else:
        metrics = data
    
    interventions = metrics.get("interventions_per_episode", 0.0)
    stability = metrics.get("stability_avg", 0.0)
    episode_length = metrics.get("avg_episode_length", 0.0)
    
    return interventions, stability, episode_length


def run_evaluation(
    mode: str,
    profile: str,
    episodes: int,
    seed: int,
    output_path: str,
    edon_gain: float = 1.0,
    edon_arch: str = "v5_heuristic"
) -> bool:
    """
    Run an evaluation and return success status.
    """
    cmd = [
        sys.executable, "run_eval.py",
        "--mode", mode,
        "--profile", profile,
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--output", output_path
    ]
    
    if mode == "edon":
        cmd.extend(["--edon-gain", str(edon_gain)])
        cmd.extend(["--edon-arch", edon_arch])
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: Evaluation failed")
        print(result.stderr)
        return False
    
    return True


def compute_improvements(
    baseline_interventions: float,
    baseline_stability: float,
    edon_interventions: float,
    edon_stability: float
) -> Tuple[float, float, float]:
    """
    Compute percentage improvements.
    
    Returns:
        (ΔInterventions%, ΔStability%, Average%)
    """
    # Lower is better for both
    delta_interventions = 100.0 * (baseline_interventions - edon_interventions) / baseline_interventions
    delta_stability = 100.0 * (baseline_stability - edon_stability) / baseline_stability
    average = (delta_interventions + delta_stability) / 2.0
    
    return delta_interventions, delta_stability, average


def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate EDON v6 learned policy")
    parser.add_argument("--profile", type=str, default="high_stress",
                       help="Stress profile to use")
    parser.add_argument("--episodes", type=int, default=30,
                       help="Number of episodes per evaluation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--edon-gain", type=float, default=1.0,
                       help="EDON gain for v6_learned")
    parser.add_argument("--baseline-output", type=str, default="results/baseline_v6_validation.json",
                       help="Output path for baseline evaluation")
    parser.add_argument("--edon-output", type=str, default="results/edon_v6_validation.json",
                       help="Output path for v6_learned evaluation")
    
    args = parser.parse_args()
    
    print("="*70)
    print("EDON v6 Validation")
    print("="*70)
    print(f"Profile: {args.profile}")
    print(f"Episodes: {args.episodes}")
    print(f"Seed: {args.seed}")
    print(f"EDON gain: {args.edon_gain}")
    print("="*70)
    
    # Run baseline
    print("\n[1/2] Running baseline evaluation...")
    if not run_evaluation(
        mode="baseline",
        profile=args.profile,
        episodes=args.episodes,
        seed=args.seed,
        output_path=args.baseline_output
    ):
        print("ERROR: Baseline evaluation failed")
        return
    
    baseline_interventions, baseline_stability, baseline_length = extract_metrics(args.baseline_output)
    print(f"Baseline results:")
    print(f"  Interventions/episode: {baseline_interventions:.2f}")
    print(f"  Stability (avg): {baseline_stability:.4f}")
    print(f"  Episode length (avg): {baseline_length:.1f} steps")
    
    # Run v6_learned
    print("\n[2/2] Running v6_learned evaluation...")
    if not run_evaluation(
        mode="edon",
        profile=args.profile,
        episodes=args.episodes,
        seed=args.seed,
        output_path=args.edon_output,
        edon_gain=args.edon_gain,
        edon_arch="v6_learned"
    ):
        print("ERROR: v6_learned evaluation failed")
        return
    
    edon_interventions, edon_stability, edon_length = extract_metrics(args.edon_output)
    print(f"v6_learned results:")
    print(f"  Interventions/episode: {edon_interventions:.2f}")
    print(f"  Stability (avg): {edon_stability:.4f}")
    print(f"  Episode length (avg): {edon_length:.1f} steps")
    
    # Compute improvements
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    delta_interventions, delta_stability, average = compute_improvements(
        baseline_interventions, baseline_stability,
        edon_interventions, edon_stability
    )
    
    print(f"ΔInterventions%: {delta_interventions:+.2f}%")
    print(f"ΔStability%: {delta_stability:+.2f}%")
    print(f"Average%: {average:+.2f}%")
    print("="*70)
    
    # Interpretation
    if average > 0:
        print(f"\n✅ v6_learned shows {average:.2f}% improvement over baseline")
    else:
        print(f"\n❌ v6_learned shows {abs(average):.2f}% degradation vs baseline")
    
    print("\nValidation complete!")


if __name__ == "__main__":
    main()

