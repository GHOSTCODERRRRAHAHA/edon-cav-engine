#!/usr/bin/env python3
"""
Simple V4.1 test - runs without modifying run_eval.py.
Uses V4.1 parameters directly in the test.
"""

import subprocess
import json
from pathlib import Path

# Test parameters
EPISODES = 30
PROFILE = "high_stress"
EDON_GAIN = 0.75
SEED = 42

RESULTS_DIR = Path("results/v41_quick")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_test(mode: str) -> str:
    """Run a test and return output file path."""
    output_file = RESULTS_DIR / f"v41_{mode}_simple.json"
    output_path = str(output_file).replace("\\", "/")
    
    cmd = [
        "python", "run_eval.py",
        "--mode", mode,
        "--episodes", str(EPISODES),
        "--profile", PROFILE,
        "--seed", str(SEED),
        "--output", output_path,
    ]
    
    if mode == "edon":
        cmd.extend(["--edon-gain", str(EDON_GAIN)])
    
    print(f"Running {mode} test...")
    print(f"  Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ERROR: Return code {result.returncode}")
        if result.stderr:
            print(f"  STDERR: {result.stderr[:500]}")
        return None
    
    if not output_file.exists():
        print(f"  WARNING: Output file not created")
        return None
    
    print(f"  [OK] Output: {output_file}")
    return str(output_file)


def compute_improvement(baseline_file: str, edon_file: str) -> dict:
    """Compute improvement metrics."""
    try:
        with open(baseline_file, "r") as f:
            baseline_data = json.load(f)
        with open(edon_file, "r") as f:
            edon_data = json.load(f)
        
        baseline_metrics = baseline_data.get("run_metrics", {})
        edon_metrics = edon_data.get("run_metrics", {})
        
        baseline_interventions = baseline_metrics.get("interventions_per_episode", 0)
        edon_interventions = edon_metrics.get("interventions_per_episode", 0)
        interventions_improvement = (
            ((baseline_interventions - edon_interventions) / baseline_interventions * 100)
            if baseline_interventions > 0 else 0.0
        )
        
        baseline_stability = baseline_metrics.get("stability_avg", 1.0)
        edon_stability = edon_metrics.get("stability_avg", 1.0)
        stability_improvement = (
            ((baseline_stability - edon_stability) / baseline_stability * 100)
            if baseline_stability > 0 else 0.0
        )
        
        average_improvement = (interventions_improvement + stability_improvement) / 2.0
        
        return {
            "interventions": interventions_improvement,
            "stability": stability_improvement,
            "average": average_improvement,
            "baseline_interventions": baseline_interventions,
            "edon_interventions": edon_interventions,
            "baseline_stability": baseline_stability,
            "edon_stability": edon_stability,
        }
    except Exception as e:
        print(f"ERROR computing improvement: {e}")
        return None


def main():
    print("="*70)
    print("V4.1 Simple Test (No File Modification)")
    print("="*70)
    print(f"Episodes: {EPISODES}")
    print(f"Profile: {PROFILE}")
    print(f"EDON gain: {EDON_GAIN}")
    print(f"Seed: {SEED}")
    print("="*70)
    
    # Check if run_eval.py exists and is valid
    if not Path("run_eval.py").exists() or Path("run_eval.py").stat().st_size == 0:
        print("\nERROR: run_eval.py is missing or corrupted!")
        print("Please restore run_eval.py before running tests.")
        return
    
    # Run baseline
    print("\n1. Running baseline test...")
    baseline_file = run_test("baseline")
    if baseline_file is None:
        print("Baseline test failed!")
        return
    
    # Run EDON
    print("\n2. Running EDON test...")
    edon_file = run_test("edon")
    if edon_file is None:
        print("EDON test failed!")
        return
    
    # Compute improvement
    print("\n3. Computing improvements...")
    improvement = compute_improvement(baseline_file, edon_file)
    
    if improvement:
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"Interventions: {improvement['interventions']:+.2f}%")
        print(f"Stability: {improvement['stability']:+.2f}%")
        print(f"Average: {improvement['average']:+.2f}%")
        print(f"\nRaw metrics:")
        print(f"  Baseline interventions: {improvement['baseline_interventions']:.1f}")
        print(f"  EDON interventions: {improvement['edon_interventions']:.1f}")
        print(f"  Baseline stability: {improvement['baseline_stability']:.4f}")
        print(f"  EDON stability: {improvement['edon_stability']:.4f}")
        
        # Save results
        results_file = RESULTS_DIR / "simple_test_results.json"
        with open(results_file, "w") as f:
            json.dump(improvement, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        
        # Status
        avg = improvement['average']
        if avg > 5.0:
            print(f"\n[EXCELLENT] {avg:+.2f}% improvement!")
        elif avg > 3.0:
            print(f"\n[GOOD] {avg:+.2f}% improvement")
        elif avg > 0.0:
            print(f"\n[POSITIVE] {avg:+.2f}% improvement (below target)")
        else:
            print(f"\n[NEGATIVE] {avg:+.2f}% (needs tuning)")


if __name__ == "__main__":
    main()

