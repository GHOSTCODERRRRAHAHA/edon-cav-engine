#!/usr/bin/env python3
"""Incremental testing to find configuration that achieves +6.4% or better."""

import json
import subprocess
from pathlib import Path

# Test matrix: vary key parameters
test_configs = [
    # Test 1: Increase base gain
    {"name": "test1_base_gain_0.5", "base_gain": 0.5, "instability": 0.4, "disturbance": 0.2, 
     "prefall_min": 0.15, "prefall_range": 0.45, "prefall_max": 0.65,
     "safe_threshold": 0.75, "safe_gain": 0.12},
    
    # Test 2: Increase instability weight
    {"name": "test2_instability_0.5", "base_gain": 0.4, "instability": 0.5, "disturbance": 0.2,
     "prefall_min": 0.15, "prefall_range": 0.45, "prefall_max": 0.65,
     "safe_threshold": 0.75, "safe_gain": 0.12},
    
    # Test 3: Increase PREFALL range
    {"name": "test3_prefall_range_0.50", "base_gain": 0.4, "instability": 0.4, "disturbance": 0.2,
     "prefall_min": 0.15, "prefall_range": 0.50, "prefall_max": 0.70,
     "safe_threshold": 0.75, "safe_gain": 0.12},
    
    # Test 4: Increase PREFALL min
    {"name": "test4_prefall_min_0.20", "base_gain": 0.4, "instability": 0.4, "disturbance": 0.2,
     "prefall_min": 0.20, "prefall_range": 0.45, "prefall_max": 0.65,
     "safe_threshold": 0.75, "safe_gain": 0.12},
    
    # Test 5: Increase both base gain and PREFALL
    {"name": "test5_base_0.45_prefall_0.50", "base_gain": 0.45, "instability": 0.4, "disturbance": 0.2,
     "prefall_min": 0.15, "prefall_range": 0.50, "prefall_max": 0.70,
     "safe_threshold": 0.75, "safe_gain": 0.12},
    
    # Test 6: Original V4 (for comparison)
    {"name": "test6_v4_original", "base_gain": 0.4, "instability": 0.4, "disturbance": 0.2,
     "prefall_min": 0.15, "prefall_range": 0.45, "prefall_max": 0.65,
     "safe_threshold": 0.75, "safe_gain": 0.12},
]

def update_config(config):
    """Update EDON_V31_HS_V4_CFG with test parameters."""
    import re
    
    config_file = "evaluation/edon_controller_v3.py"
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Replace config values
    content = re.sub(r'"BASE_GAIN":\s*[\d.]+', f'"BASE_GAIN": {config["base_gain"]}', content)
    content = re.sub(r'"INSTABILITY_WEIGHT":\s*[\d.]+', f'"INSTABILITY_WEIGHT": {config["instability"]}', content)
    content = re.sub(r'"DISTURBANCE_WEIGHT":\s*[\d.]+', f'"DISTURBANCE_WEIGHT": {config["disturbance"]}', content)
    content = re.sub(r'"PREFALL_MIN":\s*[\d.]+', f'"PREFALL_MIN": {config["prefall_min"]}', content)
    content = re.sub(r'"PREFALL_RANGE":\s*[\d.]+', f'"PREFALL_RANGE": {config["prefall_range"]}', content)
    content = re.sub(r'"PREFALL_MAX":\s*[\d.]+', f'"PREFALL_MAX": {config["prefall_max"]}', content)
    content = re.sub(r'"SAFE_THRESHOLD":\s*[\d.]+', f'"SAFE_THRESHOLD": {config["safe_threshold"]}', content)
    content = re.sub(r'"SAFE_GAIN":\s*[\d.]+', f'"SAFE_GAIN": {config["safe_gain"]}', content)
    
    with open(config_file, 'w') as f:
        f.write(content)
    
    print(f"  Updated config: BASE_GAIN={config['base_gain']}, PREFALL={config['prefall_min']}-{config['prefall_min']+config['prefall_range']}")

def run_test(config, test_num):
    """Run a single test configuration."""
    print(f"\n{'='*70}")
    print(f"Test {test_num}: {config['name']}")
    print(f"{'='*70}")
    
    # Update config file
    update_config(config)
    
    # Run baseline (if not exists)
    baseline_file = f"results/baseline_incremental_{test_num}.json"
    if not Path(baseline_file).exists():
        print(f"  Running baseline...")
        subprocess.run([
            "python", "run_eval.py",
            "--mode", "baseline",
            "--episodes", "30",
            "--profile", "high_stress",
            "--output", baseline_file
        ], capture_output=True)
    
    # Run EDON
    edon_file = f"results/edon_incremental_{test_num}_{config['name']}.json"
    print(f"  Running EDON...")
    result = subprocess.run([
        "python", "run_eval.py",
        "--mode", "edon",
        "--episodes", "30",
        "--profile", "high_stress",
        "--edon-gain", "0.75",
        "--edon-controller-version", "v3",
        "--output", edon_file
    ], capture_output=True, text=True)
    
    # Parse results
    if Path(edon_file).exists() and Path(baseline_file).exists():
        baseline = json.load(open(baseline_file))
        edon = json.load(open(edon_file))
        
        bi = baseline['interventions_per_episode']
        ei = edon['interventions_per_episode']
        bs = baseline['stability_avg']
        es = edon['stability_avg']
        
        int_imp = ((bi - ei) / bi) * 100 if bi > 0 else 0
        stab_imp = ((bs - es) / bs) * 100 if bs > 0 else 0
        avg_imp = (int_imp + stab_imp) / 2
        
        print(f"  Baseline: {bi:.1f} int/ep, stability={bs:.4f}")
        print(f"  EDON: {ei:.1f} int/ep, stability={es:.4f}")
        print(f"  Improvements: {int_imp:+.1f}% int, {stab_imp:+.1f}% stab, {avg_imp:+.1f}% avg")
        
        return {
            "name": config['name'],
            "int_imp": int_imp,
            "stab_imp": stab_imp,
            "avg_imp": avg_imp,
            "config": config
        }
    else:
        print(f"  ERROR: Results files not found")
        return None

def main():
    print("="*70)
    print("Incremental Testing - Finding Configuration for +6.4% Target")
    print("="*70)
    
    results = []
    
    for i, config in enumerate(test_configs, 1):
        result = run_test(config, i)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    for r in results:
        status = "PASS" if r['avg_imp'] >= 6.4 else "FAIL"
        print(f"{r['name']:30s} {r['avg_imp']:+6.1f}% [{status}]")
    
    # Find best
    if results:
        best = max(results, key=lambda x: x['avg_imp'])
        print(f"\nBest: {best['name']} ({best['avg_imp']:+.1f}%)")
        if best['avg_imp'] >= 6.4:
            print(f"  TARGET MET! Configuration achieves +6.4% or better")
            print(f"  Config: BASE_GAIN={best['config']['base_gain']}, "
                  f"PREFALL={best['config']['prefall_min']}-{best['config']['prefall_min']+best['config']['prefall_range']}")

if __name__ == "__main__":
    main()

