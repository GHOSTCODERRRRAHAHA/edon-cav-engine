"""
Metric-Focused Evaluation Helper for Intervention-First Optimization

Runs baseline and EDON v8 evaluations on high_stress profile and prints
a compact comparison table with explicit percentage deltas.
"""

import sys
import subprocess
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.compare_v8_vs_baseline import load_results, extract_metrics
from metrics.edon_v8_metrics import compute_episode_score_v8


def run_evaluation(mode: str, edon_arch: str = None, output_file: str = None, episodes: int = 30, seed: int = 42):
    """Run evaluation and return results."""
    cmd = [
        "python", "run_eval.py",
        "--mode", mode,
        "--profile", "high_stress",
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--output", output_file,
        "--edon-score"
    ]
    
    if mode == "edon" and edon_arch:
        cmd.extend(["--edon-gain", "1.0", "--edon-arch", edon_arch])
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running evaluation: {result.stderr}")
        return None
    
    # Load results
    if Path(output_file).exists():
        return load_results(output_file)
    return None


def compute_delta_percent(base: float, new: float) -> float:
    """Compute percentage delta."""
    if base == 0:
        if new == 0:
            return 0.0
        return float('inf') if new > 0 else float('-inf')
    return 100.0 * (new - base) / abs(base)


def main():
    """Run evaluations and print comparison."""
    import argparse
    parser = argparse.ArgumentParser(description="Metric-Focused Evaluation Helper")
    parser.add_argument("--episodes", type=int, default=30, help="Number of episodes per evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--edon-arch", type=str, default="v8_strategy", help="EDON architecture to test")
    parser.add_argument("--baseline-file", type=str, default="results/baseline_intervention_first.json", help="Baseline results file")
    parser.add_argument("--edon-file", type=str, default="results/edon_intervention_first.json", help="EDON results file")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline evaluation (use existing file)")
    parser.add_argument("--skip-edon", action="store_true", help="Skip EDON evaluation (use existing file)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Metric-Focused Evaluation: Intervention-First Optimization")
    print("="*80)
    
    # Run baseline evaluation
    baseline_data = None
    if not args.skip_baseline:
        print("\n[1/2] Running baseline evaluation...")
        baseline_data = run_evaluation("baseline", output_file=args.baseline_file, episodes=args.episodes, seed=args.seed)
    else:
        print("\n[1/2] Loading baseline results from file...")
        if Path(args.baseline_file).exists():
            baseline_data = load_results(args.baseline_file)
        else:
            print(f"Error: Baseline file not found: {args.baseline_file}")
            return
    
    # Run EDON evaluation
    edon_data = None
    if not args.skip_edon:
        print("\n[2/2] Running EDON evaluation...")
        edon_data = run_evaluation("edon", edon_arch=args.edon_arch, output_file=args.edon_file, episodes=args.episodes, seed=args.seed)
    else:
        print("\n[2/2] Loading EDON results from file...")
        if Path(args.edon_file).exists():
            edon_data = load_results(args.edon_file)
        else:
            print(f"Error: EDON file not found: {args.edon_file}")
            return
    
    if baseline_data is None or edon_data is None:
        print("Error: Failed to load evaluation results")
        return
    
    # Extract metrics
    baseline_metrics = extract_metrics(baseline_data)
    edon_metrics = extract_metrics(edon_data)
    
    # Compute EDON scores
    baseline_run_metrics = baseline_data.get("run_metrics", {})
    baseline_summary = {
        "interventions_per_episode": baseline_run_metrics.get("interventions_per_episode", 0),
        "stability_avg": baseline_run_metrics.get("stability_avg", 0.0),
        "avg_episode_length": baseline_run_metrics.get("avg_episode_length", 0)
    }
    baseline_metrics["edon_score"] = compute_episode_score_v8(baseline_summary)
    
    edon_run_metrics = edon_data.get("run_metrics", {})
    edon_summary = {
        "interventions_per_episode": edon_run_metrics.get("interventions_per_episode", 0),
        "stability_avg": edon_run_metrics.get("stability_avg", 0.0),
        "avg_episode_length": edon_run_metrics.get("avg_episode_length", 0)
    }
    edon_metrics["edon_score"] = compute_episode_score_v8(edon_summary)
    
    # Print comparison table
    print("\n" + "="*80)
    print("Comparison Table")
    print("="*80)
    print(f"{'Metric':<30} {'Baseline':<15} {'EDON':<15} {'Delta%':<15} {'Status':<15}")
    print("-"*80)
    
    # Interventions
    baseline_int = baseline_metrics.get("interventions", 0.0)
    edon_int = edon_metrics.get("interventions", 0.0)
    int_delta = compute_delta_percent(baseline_int, edon_int)
    int_status = "PASS" if int_delta <= -10.0 else "FAIL" if int_delta > 0 else "NEUTRAL"
    print(f"{'Interventions/ep':<30} {baseline_int:<15.2f} {edon_int:<15.2f} {int_delta:>+6.1f}%{'':<8} {int_status:<15}")
    
    # Stability
    baseline_stab = baseline_metrics.get("stability", 0.0)
    edon_stab = edon_metrics.get("stability", 0.0)
    stab_delta = compute_delta_percent(baseline_stab, edon_stab)
    stab_status = "PASS" if abs(stab_delta) <= 5.0 else "FAIL"
    print(f"{'Stability':<30} {baseline_stab:<15.4f} {edon_stab:<15.4f} {stab_delta:>+6.1f}%{'':<8} {stab_status:<15}")
    
    # EDON Score
    baseline_score = baseline_metrics.get("edon_score", 0.0)
    edon_score = edon_metrics.get("edon_score", 0.0)
    score_delta = compute_delta_percent(baseline_score, edon_score)
    score_status = "PASS" if score_delta > 0 else "NEUTRAL" if score_delta == 0 else "FAIL"
    print(f"{'EDON Score':<30} {baseline_score:<15.2f} {edon_score:<15.2f} {score_delta:>+6.1f}%{'':<8} {score_status:<15}")
    
    print("="*80)
    
    # Print goals check
    print("\nGoals Check:")
    print("-"*80)
    goal1_met = int_delta <= -10.0
    goal2_met = abs(stab_delta) <= 5.0
    print(f"  Goal 1: >=10% fewer interventions: {'PASS' if goal1_met else 'FAIL'} (Delta: {int_delta:.1f}%)")
    print(f"  Goal 2: Stability roughly flat (+/-5%): {'PASS' if goal2_met else 'FAIL'} (Delta: {stab_delta:.1f}%)")
    
    if goal1_met and goal2_met:
        print("\n  >>> ALL GOALS MET! <<<")
    else:
        print("\n  >>> GOALS NOT MET - Continue training/tuning <<<")
    
    print("="*80)


if __name__ == "__main__":
    main()

