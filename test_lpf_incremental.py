#!/usr/bin/env python3
"""Incremental LPF alpha formula testing."""

import json
import subprocess
from pathlib import Path
import numpy as np
import re

# Test different alpha formulas
alpha_configs = [
    {"name": "test1_alpha_075_015", "base": 0.75, "scale": 0.15, "formula": "0.75 - 0.15 * instability"},
    {"name": "test2_alpha_080_020", "base": 0.80, "scale": 0.20, "formula": "0.80 - 0.20 * instability"},
    {"name": "test3_alpha_070_010", "base": 0.70, "scale": 0.10, "formula": "0.70 - 0.10 * instability"},
    {"name": "test4_alpha_085_025", "base": 0.85, "scale": 0.25, "formula": "0.85 - 0.25 * instability"},
    {"name": "test5_alpha_072_012", "base": 0.72, "scale": 0.12, "formula": "0.72 - 0.12 * instability"},
]

N_SEEDS = 3  # Start with 3 seeds for faster iteration
EPISODES = 30
PROFILE = "high_stress"

def update_alpha_formula(config):
    """Update LPF alpha formula in edon_controller_v3.py"""
    config_file = "evaluation/edon_controller_v3.py"
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Find and replace alpha formula
    # Look for: alpha = clamp(0.8 - 0.2 * instability_score, 0.4, 0.9)
    pattern = r'alpha = clamp\([\d.]+\s*-\s*[\d.]+\s*\*\s*instability_score'
    replacement = f'alpha = clamp({config["base"]} - {config["scale"]} * instability_score'
    
    content = re.sub(pattern, replacement, content)
    
    with open(config_file, 'w') as f:
        f.write(content)
    
    print(f"  Updated alpha: {config['formula']}")

def run_test(config, test_num):
    """Run a single alpha formula test."""
    print(f"\n{'='*70}")
    print(f"Test {test_num}: {config['name']} - {config['formula']}")
    print(f"{'='*70}")
    
    # Update alpha formula
    update_alpha_formula(config)
    
    results = []
    
    for seed in range(1, N_SEEDS + 1):
        print(f"\nSeed {seed}/{N_SEEDS}...")
        
        baseline_file = f"results/baseline_lpf_test{test_num}_seed{seed}.json"
        edon_file = f"results/edon_lpf_test{test_num}_seed{seed}.json"
        
        # Run baseline (reuse if exists)
        if not Path(baseline_file).exists():
            subprocess.run([
                "python", "run_eval.py",
                "--mode", "baseline",
                "--episodes", str(EPISODES),
                "--profile", PROFILE,
                "--output", baseline_file,
                "--seed", str(seed)
            ], capture_output=True)
        
        # Run EDON
        subprocess.run([
            "python", "run_eval.py",
            "--mode", "edon",
            "--episodes", str(EPISODES),
            "--profile", PROFILE,
            "--edon-gain", "0.75",
            "--edon-controller-version", "v3",
            "--output", edon_file,
            "--seed", str(seed)
        ], capture_output=True)
        
        # Compute improvement
        if Path(baseline_file).exists() and Path(edon_file).exists():
            baseline = json.load(open(baseline_file))
            edon = json.load(open(edon_file))
            
            bi = baseline['interventions_per_episode']
            ei = edon['interventions_per_episode']
            bs = baseline['stability_avg']
            es = edon['stability_avg']
            
            int_imp = ((bi - ei) / bi) * 100 if bi > 0 else 0
            stab_imp = ((bs - es) / bs) * 100 if bs > 0 else 0
            avg_imp = (int_imp + stab_imp) / 2
            
            results.append(avg_imp)
            print(f"  {avg_imp:+.1f}%")
    
    if results:
        mean = np.mean(results)
        std = np.std(results)
        print(f"\n  Mean: {mean:+.1f}% ± {std:.1f}%")
        return {
            "name": config['name'],
            "formula": config['formula'],
            "mean": mean,
            "std": std,
            "results": results
        }
    return None

def main():
    print("="*70)
    print("Incremental LPF Alpha Formula Testing")
    print(f"{N_SEEDS} seeds × {EPISODES} episodes each")
    print("="*70)
    
    test_results = []
    
    for i, config in enumerate(alpha_configs, 1):
        result = run_test(config, i)
        if result:
            test_results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    for r in sorted(test_results, key=lambda x: x['mean'], reverse=True):
        status = "PASS" if r['mean'] >= 1.0 else "FAIL"
        print(f"{r['formula']:30s} {r['mean']:+6.1f}% ± {r['std']:.1f}% [{status}]")
    
    # Find best
    if test_results:
        best = max(test_results, key=lambda x: x['mean'])
        print(f"\nBest: {best['formula']} ({best['mean']:+.1f}% ± {best['std']:.1f}%)")
        
        if best['mean'] >= 1.0:
            print(f"  ✅ TARGET MET! LPF is helping with this formula")
        else:
            print(f"  ⚠️  Still negative, may need even lower alpha")

if __name__ == "__main__":
    main()

