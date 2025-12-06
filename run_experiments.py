#!/usr/bin/env python3
"""
Experiment Runner for EDON Humanoid Evaluation

Runs a full experiment matrix:
- Modes: baseline, edon
- Profiles: light_stress, medium_stress, high_stress
- EDON gains: 0.25, 0.5, 0.75 (for EDON mode only)

Results saved to: results/experiments/{mode}_{profile}_gain{gain}.json
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
import json


def run_evaluation(
    mode: str,
    profile: str,
    episodes: int = 20,
    seed: int = 42,
    edon_gain: float = None
) -> Tuple[bool, str]:
    """
    Run a single evaluation.
    
    Returns:
        (success: bool, output_path: str)
    """
    output_dir = Path("results/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build output filename
    if mode == "edon" and edon_gain is not None:
        output_file = f"{mode}_{profile}_gain{edon_gain:.2f}.json"
    else:
        output_file = f"{mode}_{profile}.json"
    
    output_path = output_dir / output_file
    
    # Build command
    cmd = [
        sys.executable,
        "run_eval.py",
        "--mode", mode,
        "--episodes", str(episodes),
        "--profile", profile,
        "--seed", str(seed),
        "--output", str(output_path)
    ]
    
    if mode == "edon" and edon_gain is not None:
        cmd.extend(["--edon-gain", str(edon_gain)])
    
    # Run evaluation
    print(f"\n{'='*70}")
    print(f"Running: {mode} mode, {profile} profile" + 
          (f", gain={edon_gain:.2f}" if edon_gain else ""))
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return (result.returncode == 0, str(output_path))
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Evaluation failed with exit code {e.returncode}")
        return (False, str(output_path))
    except Exception as e:
        print(f"ERROR: Failed to run evaluation: {e}")
        return (False, str(output_path))


def compute_improvements(baseline_file: str, edon_file: str) -> dict:
    """Compute percentage improvements from results."""
    try:
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
        with open(edon_file, 'r') as f:
            edon = json.load(f)
        
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
    except Exception as e:
        print(f"ERROR: Failed to compute improvements: {e}")
        return {}


def main():
    """Run full experiment matrix."""
    print("=" * 70)
    print("EDON Humanoid Evaluation - Full Experiment Matrix")
    print("=" * 70)
    print()
    print("This will run:")
    print("  - Modes: baseline, edon")
    print("  - Profiles: light_stress, medium_stress, high_stress")
    print("  - EDON gains: 0.25, 0.5, 0.75 (for EDON mode only)")
    print()
    print("Results will be saved to: results/experiments/")
    print()
    
    # Configuration
    episodes = 20
    seed = 42
    profiles = ["light_stress", "medium_stress", "high_stress"]
    edon_gains = [0.25, 0.5, 0.75]
    
    # Results tracking
    results = {}
    
    # Run baseline for each profile
    print("=" * 70)
    print("PHASE 1: Baseline Evaluations")
    print("=" * 70)
    
    for profile in profiles:
        success, output_path = run_evaluation("baseline", profile, episodes, seed)
        if success:
            results[f"baseline_{profile}"] = output_path
            print(f"✓ Baseline {profile} completed: {output_path}")
        else:
            print(f"✗ Baseline {profile} failed")
    
    # Run EDON for each profile and gain
    print("\n" + "=" * 70)
    print("PHASE 2: EDON Evaluations")
    print("=" * 70)
    
    for profile in profiles:
        for gain in edon_gains:
            success, output_path = run_evaluation("edon", profile, episodes, seed, edon_gain=gain)
            if success:
                key = f"edon_{profile}_gain{gain:.2f}"
                results[key] = output_path
                print(f"✓ EDON {profile} (gain={gain:.2f}) completed: {output_path}")
            else:
                print(f"✗ EDON {profile} (gain={gain:.2f}) failed")
    
    # Compute and print improvements
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    
    for profile in profiles:
        baseline_key = f"baseline_{profile}"
        if baseline_key not in results:
            continue
        
        print(f"\n{profile.upper()} Profile:")
        print("-" * 70)
        
        baseline_file = results[baseline_key]
        
        for gain in edon_gains:
            edon_key = f"edon_{profile}_gain{gain:.2f}"
            if edon_key not in results:
                print(f"    [SKIP] {edon_key} not found in results")
                continue
            
            edon_file = results[edon_key]
            
            # Verify file exists and is readable
            if not Path(edon_file).exists():
                print(f"    [ERROR] File not found: {edon_file}")
                continue
            
            improvements = compute_improvements(baseline_file, edon_file)
            
            if improvements:
                print(f"\n  EDON (gain={gain:.2f}) vs Baseline:")
                print(f"    Intervention reduction: {improvements.get('intervention_reduction_pct', 0.0):.1f}%")
                print(f"    Freeze reduction: {improvements.get('freeze_reduction_pct', 0.0):.1f}%")
                print(f"    Stability improvement: {improvements.get('stability_improvement_pct', 0.0):.1f}%")
                print(f"    Success rate improvement: {improvements.get('success_improvement_pct', 0.0):.1f}%")
    
    print("\n" + "=" * 70)
    print("Experiment complete!")
    print("=" * 70)
    print(f"\nAll results saved to: results/experiments/")
    print("\nTo plot specific comparisons:")
    print("  python plot_results.py --baseline results/experiments/baseline_high_stress.json \\")
    print("                          --edon results/experiments/edon_high_stress_gain0.75.json \\")
    print("                          --output plots/high_stress_strong")


if __name__ == "__main__":
    main()

