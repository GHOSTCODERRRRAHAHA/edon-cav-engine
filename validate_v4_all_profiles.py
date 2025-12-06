#!/usr/bin/env python3
"""
Multi-seed validation for V4 configuration across all stress profiles.
"""

import json
import subprocess
from pathlib import Path
import numpy as np
from typing import List, Dict

N_SEEDS = 5
EPISODES = 30
EDON_GAIN = 0.75

PROFILES = ["light_stress", "medium_stress", "high_stress", "hell_stress"]

def run_baseline(seed: int, profile: str, output_file: str) -> bool:
    """Run baseline test with specific seed."""
    result = subprocess.run([
        "python", "run_eval.py",
        "--mode", "baseline",
        "--episodes", str(EPISODES),
        "--profile", profile,
        "--output", output_file,
        "--seed", str(seed)
    ], capture_output=True, text=True)
    
    return result.returncode == 0 and Path(output_file).exists()

def run_edon(seed: int, profile: str, output_file: str) -> bool:
    """Run EDON test with specific seed."""
    result = subprocess.run([
        "python", "run_eval.py",
        "--mode", "edon",
        "--episodes", str(EPISODES),
        "--profile", profile,
        "--edon-gain", str(EDON_GAIN),
        "--edon-controller-version", "v3",
        "--output", output_file,
        "--seed", str(seed)
    ], capture_output=True, text=True)
    
    return result.returncode == 0 and Path(output_file).exists()

def compute_improvement(baseline_file: str, edon_file: str) -> Dict[str, float]:
    """Compute improvement metrics."""
    baseline = json.load(open(baseline_file))
    edon = json.load(open(edon_file))
    
    bi = baseline['interventions_per_episode']
    ei = edon['interventions_per_episode']
    bs = baseline['stability_avg']
    es = edon['stability_avg']
    
    int_imp = ((bi - ei) / bi) * 100 if bi > 0 else 0
    stab_imp = ((bs - es) / bs) * 100 if bs > 0 else 0
    avg_imp = (int_imp + stab_imp) / 2
    
    return {
        "interventions": int_imp,
        "stability": stab_imp,
        "average": avg_imp
    }

def validate_profile(profile: str):
    """Validate V4 configuration for a specific profile."""
    print(f"\n{'='*70}")
    print(f"Validating {profile.upper()} ({N_SEEDS} seeds, {EPISODES} episodes each)")
    print(f"{'='*70}")
    
    results = []
    
    for seed in range(1, N_SEEDS + 1):
        print(f"\nSeed {seed}/{N_SEEDS}...")
        
        baseline_file = f"results/baseline_v4_seed{seed}_{profile}.json"
        edon_file = f"results/edon_v4_seed{seed}_{profile}.json"
        
        if not run_baseline(seed, profile, baseline_file):
            print(f"  ERROR: Baseline failed for seed {seed}")
            continue
        
        if not run_edon(seed, profile, edon_file):
            print(f"  ERROR: EDON failed for seed {seed}")
            continue
        
        improvement = compute_improvement(baseline_file, edon_file)
        results.append(improvement)
        
        print(f"  {improvement['average']:+.1f}% avg improvement")
    
    if not results:
        print(f"\nERROR: No valid results for {profile}")
        return None
    
    # Compute statistics
    avg_imps = [r['average'] for r in results]
    avg_mean = np.mean(avg_imps)
    avg_std = np.std(avg_imps)
    
    label = f"{profile.replace('_', ' ').title()} V4: ~{avg_mean:.1f}% ± {avg_std:.1f}% avg improvement"
    
    print(f"\n{label}")
    print(f"  Range: {min(avg_imps):+.1f}% to {max(avg_imps):+.1f}%")
    
    return {
        "profile": profile,
        "mean": float(avg_mean),
        "std": float(avg_std),
        "min": float(min(avg_imps)),
        "max": float(max(avg_imps)),
        "label": label,
        "n_seeds": len(results)
    }

def main():
    print("="*70)
    print("V4 Multi-Seed Validation - All Profiles")
    print("="*70)
    
    all_results = {}
    
    for profile in PROFILES:
        result = validate_profile(profile)
        if result:
            all_results[profile] = result
    
    # Summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    
    for profile, result in all_results.items():
        print(f"\n{result['label']}")
    
    # Save summary
    summary_file = "results/v4_validation_all_profiles_summary.json"
    json.dump(all_results, open(summary_file, 'w'), indent=2)
    print(f"\n\nSummary saved to: {summary_file}")

if __name__ == "__main__":
    main()

