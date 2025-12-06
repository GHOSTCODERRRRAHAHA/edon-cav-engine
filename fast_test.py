#!/usr/bin/env python3
"""
Fast test mode for quick EDON performance validation.
Runs 8-10 episodes, 1 seed, and provides instant improvement summary.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


def run_baseline(episodes: int, profile: str, seed: int, output_file: str) -> bool:
    """Run baseline test."""
    print(f"Running baseline ({episodes} episodes, seed {seed})...")
    cmd = [
        "python", "run_eval.py",
        "--mode", "baseline",
        "--profile", profile,  # Force profile parameter
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--output", output_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print debug output if present
    if "[RUN-EVAL]" in result.stdout:
        print(result.stdout.split("[RUN-EVAL]")[1].split("\n")[0])
    elif "[RUN-EVAL]" in result.stderr:
        print(result.stderr.split("[RUN-EVAL]")[1].split("\n")[0])
    
    if result.returncode != 0:
        print(f"ERROR: Baseline failed")
        print(result.stderr[-500:])
        return False
    
    return Path(output_file).exists()


def run_edon(episodes: int, profile: str, seed: int, gain: float, output_file: str) -> bool:
    """Run EDON test."""
    print(f"Running EDON ({episodes} episodes, seed {seed}, gain {gain})...")
    cmd = [
        "python", "run_eval.py",
        "--mode", "edon",
        "--profile", profile,  # Force profile parameter
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--edon-gain", str(gain),  # Force gain parameter
        "--output", output_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print debug output if present
    if "[RUN-EVAL]" in result.stdout:
        print(result.stdout.split("[RUN-EVAL]")[1].split("\n")[0])
    elif "[RUN-EVAL]" in result.stderr:
        print(result.stderr.split("[RUN-EVAL]")[1].split("\n")[0])
    
    if result.returncode != 0:
        print(f"ERROR: EDON failed")
        print(result.stderr[-500:])
        return False
    
    return Path(output_file).exists()


def load_results(json_file: str) -> Optional[Dict]:
    """Load results from JSON file."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load {json_file}: {e}")
        return None


def compute_improvements(baseline_data: Dict, edon_data: Dict) -> Dict[str, float]:
    """Compute improvement percentages."""
    baseline_metrics = baseline_data.get("run_metrics", {})
    edon_metrics = edon_data.get("run_metrics", {})
    
    baseline_int = baseline_metrics.get("interventions_per_episode", 0.0)
    edon_int = edon_metrics.get("interventions_per_episode", 0.0)
    baseline_stab = baseline_metrics.get("stability_avg", 0.0)
    edon_stab = edon_metrics.get("stability_avg", 0.0)
    
    # Interventions: lower is better
    if baseline_int > 0:
        interv_improvement = ((baseline_int - edon_int) / baseline_int) * 100.0
    else:
        interv_improvement = 0.0
    
    # Stability: lower is better (stability_avg is variance, so lower = more stable)
    if baseline_stab > 0:
        stab_improvement = ((baseline_stab - edon_stab) / baseline_stab) * 100.0
    else:
        stab_improvement = 0.0
    
    average = (interv_improvement + stab_improvement) / 2.0
    
    return {
        "baseline_int": baseline_int,
        "edon_int": edon_int,
        "baseline_stab": baseline_stab,
        "edon_stab": edon_stab,
        "interv_improvement": interv_improvement,
        "stab_improvement": stab_improvement,
        "average": average
    }


def print_summary(episodes: int, profile: str, gain: float, results: Dict[str, float]):
    """Print clean summary."""
    print(f"\nFAST TEST ({episodes} episodes):")
    print()
    print(f"Baseline: interventions={results['baseline_int']:.1f}  stability={results['baseline_stab']:.4f}")
    print(f"EDON:     interventions={results['edon_int']:.1f}  stability={results['edon_stab']:.4f}")
    print()
    print("IMPROVEMENT:")
    print(f"- Interventions: {results['interv_improvement']:+.1f}%")
    print(f"- Stability:     {results['stab_improvement']:+.1f}%")
    
    avg = results['average']
    status = "KEEP" if avg > 0 else "REJECT"
    print(f"- Average:       {avg:+.1f}%  ({status})")


def save_results(profile: str, gain: float, results: Dict[str, float], episodes: int, seed: int):
    """Save results to JSON file."""
    output_dir = Path("results/fast_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"fast_{profile}_g{gain:.2f}.json"
    
    data = {
        "profile": profile,
        "gain": gain,
        "episodes": episodes,
        "seed": seed,
        "baseline": {
            "interventions_per_episode": results["baseline_int"],
            "stability_avg": results["baseline_stab"]
        },
        "edon": {
            "interventions_per_episode": results["edon_int"],
            "stability_avg": results["edon_stab"]
        },
        "improvements": {
            "interventions": results["interv_improvement"],
            "stability": results["stab_improvement"],
            "average": results["average"]
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Fast test mode for EDON performance validation")
    parser.add_argument("--gain", type=float, required=True, help="EDON gain value")
    parser.add_argument("--profile", type=str, default="high_stress", help="Stress profile (default: high_stress)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # Create temp output directory
    temp_dir = Path("results/fast_tests")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_file = str(temp_dir / "tmp_baseline.json")
    edon_file = str(temp_dir / "tmp_edon.json")
    
    # Run baseline
    if not run_baseline(args.episodes, args.profile, args.seed, baseline_file):
        print("ERROR: Baseline test failed")
        sys.exit(1)
    
    # Run EDON
    if not run_edon(args.episodes, args.profile, args.seed, args.gain, edon_file):
        print("ERROR: EDON test failed")
        sys.exit(1)
    
    # Load results
    baseline_data = load_results(baseline_file)
    edon_data = load_results(edon_file)
    
    if baseline_data is None or edon_data is None:
        print("ERROR: Failed to load results")
        sys.exit(1)
    
    # Compute improvements
    results = compute_improvements(baseline_data, edon_data)
    
    # Print summary
    print_summary(args.episodes, args.profile, args.gain, results)
    
    # Save results
    save_results(args.profile, args.gain, results, args.episodes, args.seed)
    
    # Clean up temp files
    try:
        Path(baseline_file).unlink()
        Path(edon_file).unlink()
    except:
        pass


if __name__ == "__main__":
    main()

