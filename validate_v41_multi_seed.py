#!/usr/bin/env python3
"""
Multi-seed validation for V4.1 configuration (10% target).
Runs N_SEEDS tests and computes mean ± std improvement.
"""

import json
import subprocess
from pathlib import Path
import numpy as np
from typing import List, Dict

N_SEEDS = 5
EPISODES = 30
PROFILE = "high_stress"  # Testing high stress for 10% target
EDON_GAIN = 0.75

def run_baseline(seed: int, output_file: str) -> bool:
    """Run baseline test with specific seed."""
    print(f"  Running baseline (seed {seed})...")
    result = subprocess.run([
        "python", "run_eval.py",
        "--mode", "baseline",
        "--episodes", str(EPISODES),
        "--profile", PROFILE,
        "--output", output_file,
        "--seed", str(seed)
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ERROR: Baseline failed: {result.stderr[-200:]}")
        return False
    
    return Path(output_file).exists()

def run_edon(seed: int, output_file: str) -> bool:
    """Run EDON test with specific seed."""
    print(f"  Running EDON (seed {seed})...")
    result = subprocess.run([
        "python", "run_eval.py",
        "--mode", "edon",
        "--episodes", str(EPISODES),
        "--profile", PROFILE,
        "--edon-gain", str(EDON_GAIN),
        "--output", output_file,
        "--seed", str(seed)
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ERROR: EDON failed: {result.stderr[-200:]}")
        return False
    
    return Path(output_file).exists()

def compute_improvement(baseline_file: str, edon_file: str) -> Dict[str, float]:
    """Compute improvement metrics."""
    baseline = json.load(open(baseline_file))
    edon = json.load(open(edon_file))
    
    # Handle both old and new result formats
    if 'interventions_per_episode' in baseline:
        bi = baseline['interventions_per_episode']
        ei = edon['interventions_per_episode']
        bs = baseline['stability_avg']
        es = edon['stability_avg']
    elif 'run_metrics' in baseline:
        bi = baseline['run_metrics'].get('interventions', 0)
        ei = edon['run_metrics'].get('interventions', 0)
        bs = baseline['run_metrics'].get('stability_score', 1.0)
        es = edon['run_metrics'].get('stability_score', 1.0)
    else:
        # Fallback
        bi = baseline.get('interventions', 0)
        ei = edon.get('interventions', 0)
        bs = baseline.get('stability_score', 1.0)
        es = edon.get('stability_score', 1.0)
    
    int_imp = ((bi - ei) / bi) * 100 if bi > 0 else 0
    stab_imp = ((bs - es) / bs) * 100 if bs > 0 else 0
    avg_imp = (int_imp + stab_imp) / 2
    
    return {
        "interventions": int_imp,
        "stability": stab_imp,
        "average": avg_imp,
        "baseline_int": bi,
        "edon_int": ei,
        "baseline_stab": bs,
        "edon_stab": es
    }

def main():
    print("="*70)
    print(f"V4.1 Multi-Seed Validation ({N_SEEDS} seeds, {EPISODES} episodes each)")
    print(f"Profile: {PROFILE}")
    print(f"Configuration: Kp=0.15, Kd=0.04, BASE_GAIN=0.53, PREFALL=0.18-0.65")
    print("="*70)
    
    results = []
    
    for seed in range(1, N_SEEDS + 1):
        print(f"\n{'='*70}")
        print(f"Seed {seed}/{N_SEEDS}")
        print(f"{'='*70}")
        
        baseline_file = f"results/baseline_v41_seed{seed}_{PROFILE}.json"
        edon_file = f"results/edon_v41_seed{seed}_{PROFILE}.json"
        
        # Run baseline
        if not run_baseline(seed, baseline_file):
            print(f"  ERROR: Baseline failed for seed {seed}")
            continue
        
        # Run EDON
        if not run_edon(seed, edon_file):
            print(f"  ERROR: EDON failed for seed {seed}")
            continue
        
        # Compute improvement
        improvement = compute_improvement(baseline_file, edon_file)
        results.append(improvement)
        
        print(f"  Seed {seed} results:")
        print(f"    Interventions: {improvement['interventions']:+.1f}%")
        print(f"    Stability: {improvement['stability']:+.1f}%")
        print(f"    Average: {improvement['average']:+.1f}%")
    
    # Summary statistics
    if not results:
        print("\nERROR: No valid results collected")
        return
    
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    
    int_imps = [r['interventions'] for r in results]
    stab_imps = [r['stability'] for r in results]
    avg_imps = [r['average'] for r in results]
    
    int_mean = np.mean(int_imps)
    int_std = np.std(int_imps)
    stab_mean = np.mean(stab_imps)
    stab_std = np.std(stab_imps)
    avg_mean = np.mean(avg_imps)
    avg_std = np.std(avg_imps)
    
    print(f"\nInterventions:")
    print(f"  Mean: {int_mean:+.1f}%")
    print(f"  Std:  {int_std:.1f}%")
    print(f"  Range: {min(int_imps):+.1f}% to {max(int_imps):+.1f}%")
    
    print(f"\nStability:")
    print(f"  Mean: {stab_mean:+.1f}%")
    print(f"  Std:  {stab_std:.1f}%")
    print(f"  Range: {min(stab_imps):+.1f}% to {max(stab_imps):+.1f}%")
    
    print(f"\nAverage Improvement:")
    print(f"  Mean: {avg_mean:+.1f}%")
    print(f"  Std:  {avg_std:.1f}%")
    print(f"  Range: {min(avg_imps):+.1f}% to {max(avg_imps):+.1f}%")
    
    print(f"\n{'='*70}")
    print(f"CONFIGURATION LABEL")
    print(f"{'='*70}")
    print(f"\n{PROFILE.capitalize().replace('_', ' ')} V4.1: ~{avg_mean:.1f}% ± {avg_std:.1f}% avg improvement")
    print(f"  (Based on {N_SEEDS} seeds, {EPISODES} episodes each)")
    print(f"\nIndividual seed results:")
    for i, r in enumerate(results, 1):
        print(f"  Seed {i}: {r['average']:+.1f}% (Int: {r['interventions']:+.1f}%, Stab: {r['stability']:+.1f}%)")
    
    # Save summary
    summary_file = f"results/v41_validation_{PROFILE}_summary.json"
    summary = {
        "version": "V4.1",
        "profile": PROFILE,
        "n_seeds": N_SEEDS,
        "episodes_per_seed": EPISODES,
        "edon_gain": EDON_GAIN,
        "configuration": {
            "EDON_BASE_KP_ROLL": 0.15,
            "EDON_BASE_KP_PITCH": 0.15,
            "EDON_BASE_KD_ROLL": 0.04,
            "EDON_BASE_KD_PITCH": 0.04,
            "BASE_GAIN": 0.53,
            "PREFALL_MIN": 0.18,
            "PREFALL_MAX": 0.65,
            "RECOVERY_BOOST_FAIL": 1.22,
            "RECOVERY_BOOST_PREFALL": 1.12,
        },
        "interventions": {
            "mean": float(int_mean),
            "std": float(int_std),
            "min": float(min(int_imps)),
            "max": float(max(int_imps)),
            "values": [float(x) for x in int_imps]
        },
        "stability": {
            "mean": float(stab_mean),
            "std": float(stab_std),
            "min": float(min(stab_imps)),
            "max": float(max(stab_imps)),
            "values": [float(x) for x in stab_imps]
        },
        "average": {
            "mean": float(avg_mean),
            "std": float(avg_std),
            "min": float(min(avg_imps)),
            "max": float(max(avg_imps)),
            "values": [float(x) for x in avg_imps]
        },
        "label": f"{PROFILE.capitalize().replace('_', ' ')} V4.1: ~{avg_mean:.1f}% ± {avg_std:.1f}% avg improvement",
        "individual_results": [
            {
                "seed": i+1,
                "interventions": float(r['interventions']),
                "stability": float(r['stability']),
                "average": float(r['average'])
            }
            for i, r in enumerate(results)
        ]
    }
    
    json.dump(summary, open(summary_file, 'w'), indent=2)
    print(f"\nSummary saved to: {summary_file}")
    
    return summary

if __name__ == "__main__":
    main()


