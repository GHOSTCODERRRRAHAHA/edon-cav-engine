#!/usr/bin/env python3
"""Extended incremental testing with more parameter combinations."""

import json
import subprocess
from pathlib import Path
import re

# Extended test matrix
test_configs = [
    # Test 1: BASE_GAIN=0.5 (already tested, got +7.0%)
    {"name": "test1_base_0.5", "base_gain": 0.5, "instability": 0.4, "disturbance": 0.2,
     "prefall_min": 0.15, "prefall_range": 0.45, "prefall_max": 0.65,
     "safe_threshold": 0.75, "safe_gain": 0.12},
    
    # Test 7: BASE_GAIN=0.5 with higher instability weight
    {"name": "test7_base_0.5_inst_0.5", "base_gain": 0.5, "instability": 0.5, "disturbance": 0.2,
     "prefall_min": 0.15, "prefall_range": 0.45, "prefall_max": 0.65,
     "safe_threshold": 0.75, "safe_gain": 0.12},
    
    # Test 8: BASE_GAIN=0.5 with higher PREFALL range
    {"name": "test8_base_0.5_prefall_0.50", "base_gain": 0.5, "instability": 0.4, "disturbance": 0.2,
     "prefall_min": 0.15, "prefall_range": 0.50, "prefall_max": 0.70,
     "safe_threshold": 0.75, "safe_gain": 0.12},
    
    # Test 9: BASE_GAIN=0.5 with lower PREFALL min (less aggressive)
    {"name": "test9_base_0.5_prefall_min_0.10", "base_gain": 0.5, "instability": 0.4, "disturbance": 0.2,
     "prefall_min": 0.10, "prefall_range": 0.45, "prefall_max": 0.65,
     "safe_threshold": 0.75, "safe_gain": 0.12},
    
    # Test 10: BASE_GAIN=0.55 (slightly higher)
    {"name": "test10_base_0.55", "base_gain": 0.55, "instability": 0.4, "disturbance": 0.2,
     "prefall_min": 0.15, "prefall_range": 0.45, "prefall_max": 0.65,
     "safe_threshold": 0.75, "safe_gain": 0.12},
    
    # Test 11: BASE_GAIN=0.5 with recovery boost increase
    {"name": "test11_base_0.5_recovery_1.3", "base_gain": 0.5, "instability": 0.4, "disturbance": 0.2,
     "prefall_min": 0.15, "prefall_range": 0.45, "prefall_max": 0.65,
     "safe_threshold": 0.75, "safe_gain": 0.12, "recovery_boost": 1.3},
]

def update_config(config):
    """Update EDON_V31_HS_V4_CFG with test parameters."""
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
    if "recovery_boost" in config:
        content = re.sub(r'"RECOVERY_BOOST":\s*[\d.]+', f'"RECOVERY_BOOST": {config["recovery_boost"]}', content)
    
    with open(config_file, 'w') as f:
        f.write(content)

def run_test(config, test_num):
    """Run a single test configuration."""
    print(f"\nTest {test_num}: {config['name']}")
    
    # Update config file
    update_config(config)
    
    # Run EDON (reuse baseline from test 1)
    baseline_file = "results/baseline_incremental_1.json"
    edon_file = f"results/edon_incremental_{test_num}_{config['name']}.json"
    
    print(f"  Running EDON...")
    subprocess.run([
        "python", "run_eval.py",
        "--mode", "edon",
        "--episodes", "30",
        "--profile", "high_stress",
        "--edon-gain", "0.75",
        "--edon-controller-version", "v3",
        "--output", edon_file
    ], capture_output=True)
    
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
        
        print(f"  {int_imp:+.1f}% int, {stab_imp:+.1f}% stab, {avg_imp:+.1f}% avg")
        
        return {
            "name": config['name'],
            "int_imp": int_imp,
            "stab_imp": stab_imp,
            "avg_imp": avg_imp,
            "config": config
        }
    return None

def main():
    print("="*70)
    print("Extended Incremental Testing")
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
    
    for r in sorted(results, key=lambda x: x['avg_imp'], reverse=True):
        status = "PASS" if r['avg_imp'] >= 6.4 else "FAIL"
        print(f"{r['name']:30s} {r['avg_imp']:+6.1f}% [{status}]")
    
    # Find best
    if results:
        best = max(results, key=lambda x: x['avg_imp'])
        print(f"\nBest: {best['name']} ({best['avg_imp']:+.1f}%)")
        if best['avg_imp'] >= 6.4:
            print(f"  TARGET MET!")

if __name__ == "__main__":
    main()

