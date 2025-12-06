"""
Compare EDON v7 (200 episodes, bold PPO) vs Baseline.

Loads results from:
- results/baseline_ep200_bold.json
- results/edon_v7_ep200_bold.json

Prints comparison and verdict.
"""

import json
import sys
from pathlib import Path

# Import EDON score computation
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.edon_score import compute_episode_score


def load_results(filepath: str) -> dict:
    """Load JSON results file."""
    path = Path(filepath)
    if not path.exists():
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(results: dict) -> dict:
    """Extract key metrics from results."""
    # Try run_metrics first (new format)
    run_metrics = results.get("run_metrics", {})
    if isinstance(run_metrics, dict):
        interventions = run_metrics.get("interventions_per_episode", 0)
        stability = run_metrics.get("stability_avg", 0.0)
        episode_length = run_metrics.get("avg_episode_length", 0.0)
    else:
        # Fallback to summary (old format)
        summary = results.get("summary", {})
        interventions = summary.get("interventions_per_episode", 0)
        stability = summary.get("avg_stability", 0.0)
        episode_length = summary.get("avg_episode_length", 0.0)
    
    # Compute EDON score from components (not stored in JSON)
    episode_summary = {
        "interventions_per_episode": interventions,
        "stability_avg": stability,
        "avg_episode_length": episode_length
    }
    edon_score = compute_episode_score(episode_summary)
    
    return {
        "interventions": float(interventions) if interventions else 0.0,
        "stability": float(stability) if stability else 0.0,
        "edon_score": float(edon_score) if edon_score else 0.0,
        "episode_length": float(episode_length) if episode_length else 0.0
    }


def compute_delta(base: float, new: float) -> float:
    """Compute percentage delta: 100 * (base - new) / base."""
    if base == 0:
        return 0.0
    return 100.0 * (base - new) / base


def main():
    """Main comparison."""
    print("="*70)
    print("EDON v7 (200 episodes, bold PPO) vs Baseline Comparison")
    print("="*70)
    print()
    
    # Load results
    baseline_results = load_results("results/baseline_ep200_bold.json")
    v7_results = load_results("results/edon_v7_ep200_bold.json")
    
    # Extract metrics
    baseline_metrics = extract_metrics(baseline_results)
    v7_metrics = extract_metrics(v7_results)
    
    # Print baseline
    print("Baseline:")
    print(f"  Interventions: {baseline_metrics['interventions']:.2f}")
    print(f"  Stability: {baseline_metrics['stability']:.4f}")
    print(f"  EDON Score: {baseline_metrics['edon_score']:.2f}")
    print(f"  Episode Length: {baseline_metrics['episode_length']:.1f}")
    print()
    
    # Print v7
    print("EDON v7 (ep200_bold):")
    print(f"  Interventions: {v7_metrics['interventions']:.2f}")
    print(f"  Stability: {v7_metrics['stability']:.4f}")
    print(f"  EDON Score: {v7_metrics['edon_score']:.2f}")
    print(f"  Episode Length: {v7_metrics['episode_length']:.1f}")
    print()
    
    # Compute deltas
    delta_interventions = compute_delta(
        baseline_metrics['interventions'],
        v7_metrics['interventions']
    )
    delta_stability = compute_delta(
        baseline_metrics['stability'],
        v7_metrics['stability']
    )
    delta_edon_score = v7_metrics['edon_score'] - baseline_metrics['edon_score']
    delta_interventions_abs = baseline_metrics['interventions'] - v7_metrics['interventions']
    
    print("Deltas:")
    print(f"  DeltaInterventions%: {delta_interventions:+.2f}%")
    print(f"  DeltaStability%: {delta_stability:+.2f}%")
    print(f"  DeltaEDON Score: {delta_edon_score:+.2f}")
    print(f"  DeltaInterventions (abs): {delta_interventions_abs:+.2f} per episode")
    print()
    
    # Verdict
    print("="*70)
    print("VERDICT")
    print("="*70)
    
    # Success criteria:
    # 1. EDON Score improves by ≥ +1.0, OR
    # 2. Interventions drop by ≥ 2 per episode, with stability not worse
    
    score_improved = delta_edon_score >= 1.0
    interventions_dropped = delta_interventions_abs >= 2.0
    stability_not_worse = delta_stability >= 0  # Stability improved or same
    
    if score_improved:
        verdict = "[PASS] V7 shows meaningful improvement over baseline"
        print(verdict)
        print(f"   - EDON Score improved by {delta_edon_score:.2f} points (threshold: +1.0)")
    elif interventions_dropped and stability_not_worse:
        verdict = "[PASS] V7 shows meaningful improvement over baseline"
        print(verdict)
        print(f"   - Interventions dropped by {delta_interventions_abs:.2f} per episode (threshold: 2.0)")
        print(f"   - Stability change: {delta_stability:+.2f}% (not worse)")
    elif delta_edon_score <= -1.0:
        verdict = "[REGRESS] V7 is worse than baseline"
        print(verdict)
        print(f"   - EDON Score decreased by {abs(delta_edon_score):.2f} points")
    else:
        verdict = "[NEUTRAL] No meaningful improvement"
        print(verdict)
        print(f"   - EDON Score change: {delta_edon_score:+.2f} (threshold: +1.0)")
        print(f"   - Interventions change: {delta_interventions_abs:+.2f} per episode (threshold: 2.0)")
        print(f"   - Stability change: {delta_stability:+.2f}%")
        print()
        print("   Analysis: Performance change is below success thresholds.")
        print("   Possible next steps:")
        print("   - Train longer (500+ episodes)")
        print("   - Further tune reward function")
        print("   - Increase learning rate or update epochs")
    
    print()


if __name__ == "__main__":
    main()

