#!/usr/bin/env python3
"""Incremental testing for predicted boost parameters."""

import json
import subprocess
from pathlib import Path
import numpy as np
import re

# Test different predicted boost configurations
boost_configs = [
    # Test 1: Very conservative (0.15 multiplier, 1.5 scale)
    {"name": "test1_boost_015_scale_15", "boost_mult": 0.15, "scale": 1.5, "cap": 0.5},
    
    # Test 2: Current V5.2 (0.2 multiplier, 2.0 scale)
    {"name": "test2_boost_020_scale_20", "boost_mult": 0.20, "scale": 2.0, "cap": 0.5},
    
    # Test 3: Slightly more (0.25 multiplier, 2.5 scale)
    {"name": "test3_boost_025_scale_25", "boost_mult": 0.25, "scale": 2.5, "cap": 0.5},
    
    # Test 4: Lower scale, same multiplier (0.2 multiplier, 1.5 scale)
    {"name": "test4_boost_020_scale_15", "boost_mult": 0.20, "scale": 1.5, "cap": 0.5},
    
    # Test 5: Higher cap (0.2 multiplier, 2.0 scale, 0.6 cap)
    {"name": "test5_boost_020_scale_20_cap_06", "boost_mult": 0.20, "scale": 2.0, "cap": 0.6},
]

N_SEEDS = 3  # Start with 3 seeds for faster iteration
EPISODES = 30
PROFILE = "medium_stress"

def update_predicted_boost(config):
    """Update predicted boost parameters in edon_controller_v3.py"""
    config_file = "evaluation/edon_controller_v3.py"
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Update predicted instability calculation
    # Find: predicted_instability = max(0.0, min(delta_ema * 2.0, 0.5))
    pattern1 = r'predicted_instability = max\(0\.0, min\(delta_ema \* [\d.]+, [\d.]+\)\)'
    replacement1 = f'predicted_instability = max(0.0, min(delta_ema * {config["scale"]}, {config["cap"]}))'
    content = re.sub(pattern1, replacement1, content)
    
    # Update gain modulation
    # Find: gain *= (1.0 + 0.2 * predicted_instability)
    pattern2 = r'gain \*= \(1\.0 \+ [\d.]+ \* predicted_instability\)'
    replacement2 = f'gain *= (1.0 + {config["boost_mult"]} * predicted_instability)'
    content = re.sub(pattern2, replacement2, content)
    
    with open(config_file, 'w') as f:
        f.write(content)
    
    print(f"  Updated: scale={config['scale']}, cap={config['cap']}, boost_mult={config['boost_mult']}")

def run_test(config, test_num):
    """Run a single boost configuration test."""
    print(f"\n{'='*70}")
    print(f"Test {test_num}: {config['name']}")
    print(f"  Boost mult: {config['boost_mult']}, Scale: {config['scale']}, Cap: {config['cap']}")
    print(f"{'='*70}")
    
    # Update config
    update_predicted_boost(config)
    
    results = []
    
    for seed in range(1, N_SEEDS + 1):
        print(f"\nSeed {seed}/{N_SEEDS}...")
        
        baseline_file = f"results/baseline_boost_test{test_num}_seed{seed}_{PROFILE}.json"
        edon_file = f"results/edon_boost_test{test_num}_seed{seed}_{PROFILE}.json"
        
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
            "boost_mult": config['boost_mult'],
            "scale": config['scale'],
            "cap": config['cap'],
            "mean": mean,
            "std": std,
            "results": results
        }
    return None

def main():
    print("="*70)
    print("Incremental Predicted Boost Testing")
    print(f"Profile: {PROFILE}, {N_SEEDS} seeds × {EPISODES} episodes each")
    print("="*70)
    
    test_results = []
    
    for i, config in enumerate(boost_configs, 1):
        result = run_test(config, i)
        if result:
            test_results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    for r in sorted(test_results, key=lambda x: x['mean'], reverse=True):
        status = "PASS" if 1.0 <= r['mean'] <= 3.0 else "CLOSE" if r['mean'] > 0 else "FAIL"
        print(f"Boost {r['boost_mult']:.2f}, Scale {r['scale']:.1f}, Cap {r['cap']:.1f}: "
              f"{r['mean']:+6.1f}% ± {r['std']:.1f}% [{status}]")
    
    # Find best
    if test_results:
        best = max(test_results, key=lambda x: x['mean'])
        print(f"\nBest: Boost {best['boost_mult']:.2f}, Scale {best['scale']:.1f}, Cap {best['cap']:.1f}")
        print(f"  Result: {best['mean']:+.1f}% ± {best['std']:.1f}%")
        
        if 1.0 <= best['mean'] <= 3.0:
            print(f"  ✅ TARGET MET! (+1% to +3%)")
        elif best['mean'] > 0:
            print(f"  ⚠️  Positive but below target, may need slight increase")
        else:
            print(f"  ❌ Still negative, may need different approach")

if __name__ == "__main__":
    main()

