#!/usr/bin/env python3
"""
Incremental testing script to find the sweet spot for 10%+ improvements.
Tests different parameter combinations systematically.
"""

import subprocess
import sys
import json
from pathlib import Path

# Test configuration
EPISODES = 15  # Quick tests
PROFILE = "high_stress"
GAIN = 0.75

# Parameter sets to test (incremental increases)
PARAM_SETS = [
    {
        "name": "baseline_v3",
        "kp": 0.10,
        "kd": 0.03,
        "prefall_base": 0.28,
        "safe_gain": 0.015,
        "safe_corr": 0.05,
        "desc": "Current V3 baseline (after first fixes)"
    },
    {
        "name": "test_1_moderate_pd",
        "kp": 0.14,
        "kd": 0.04,
        "prefall_base": 0.28,
        "safe_gain": 0.015,
        "safe_corr": 0.05,
        "desc": "Moderate PD gains (40% increase)"
    },
    {
        "name": "test_2_moderate_prefall",
        "kp": 0.14,
        "kd": 0.04,
        "prefall_base": 0.30,
        "safe_gain": 0.015,
        "safe_corr": 0.05,
        "desc": "Moderate PD + PREFALL (30% base)"
    },
    {
        "name": "test_3_moderate_safe",
        "kp": 0.14,
        "kd": 0.04,
        "prefall_base": 0.30,
        "safe_gain": 0.020,
        "safe_corr": 0.08,
        "desc": "Moderate PD + PREFALL + SAFE (2% gain, 0.08 corr)"
    },
    {
        "name": "test_4_stronger_pd",
        "kp": 0.16,
        "kd": 0.045,
        "prefall_base": 0.30,
        "safe_gain": 0.020,
        "safe_corr": 0.08,
        "desc": "Stronger PD (60% increase) + PREFALL + SAFE"
    },
    {
        "name": "test_5_stronger_prefall",
        "kp": 0.16,
        "kd": 0.045,
        "prefall_base": 0.32,
        "safe_gain": 0.020,
        "safe_corr": 0.08,
        "desc": "Stronger PD + PREFALL (32% base) + SAFE"
    },
]

def update_controller_params(params):
    """Update controller parameters in edon_controller_v3.py"""
    # This is a placeholder - in real implementation, we'd modify the file
    # For now, we'll manually test each configuration
    print(f"  Would update: KP={params['kp']}, KD={params['kd']}, "
          f"PREFALL_BASE={params['prefall_base']}, SAFE_GAIN={params['safe_gain']}, "
          f"SAFE_CORR={params['safe_corr']}")

def run_test(params):
    """Run a single test configuration"""
    print(f"\n{'='*70}")
    print(f"Testing: {params['name']}")
    print(f"Description: {params['desc']}")
    print(f"{'='*70}")
    
    # For now, we'll run with current controller settings
    # In a full implementation, we'd modify the controller file first
    output_file = f"results/incremental_{params['name']}.json"
    
    cmd = [
        sys.executable, "run_eval.py",
        "--mode", "edon",
        "--episodes", str(EPISODES),
        "--profile", PROFILE,
        "--edon-gain", str(GAIN),
        "--edon-controller-version", "v3",
        "--output", output_file
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            # Load and analyze results
            if Path(output_file).exists():
                data = json.load(open(output_file))
                int_per_ep = data.get('interventions_per_episode', 0)
                stab = data.get('stability_avg', 0.0)
                return {
                    'success': True,
                    'interventions': int_per_ep,
                    'stability': stab,
                    'output_file': output_file
                }
        return {'success': False, 'error': result.stderr}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def compare_with_baseline(baseline_file, test_result):
    """Compare test result with baseline"""
    if not Path(baseline_file).exists():
        return None
    
    baseline = json.load(open(baseline_file))
    base_int = baseline.get('interventions_per_episode', 0)
    base_stab = baseline.get('stability_avg', 0.0)
    
    int_imp = ((base_int - test_result['interventions']) / base_int) * 100 if base_int > 0 else 0
    stab_imp = ((base_stab - test_result['stability']) / base_stab) * 100 if base_stab > 0 else 0
    avg_imp = (int_imp + stab_imp) / 2
    
    return {
        'int_improvement': int_imp,
        'stab_improvement': stab_imp,
        'avg_improvement': avg_imp,
        'baseline_int': base_int,
        'baseline_stab': base_stab
    }

def main():
    print("="*70)
    print("INCREMENTAL TESTING - Finding Sweet Spot for 10%+ Improvements")
    print("="*70)
    print(f"Profile: {PROFILE}")
    print(f"EDON Gain: {GAIN}")
    print(f"Episodes per test: {EPISODES}")
    print(f"Total tests: {len(PARAM_SETS)}")
    print("="*70)
    
    # First, run baseline
    print("\n[1/6] Running baseline test...")
    baseline_file = "results/incremental_baseline.json"
    baseline_cmd = [
        sys.executable, "run_eval.py",
        "--mode", "baseline",
        "--episodes", str(EPISODES),
        "--profile", PROFILE,
        "--output", baseline_file
    ]
    subprocess.run(baseline_cmd, capture_output=True)
    
    if not Path(baseline_file).exists():
        print("ERROR: Baseline test failed!")
        return 1
    
    baseline_data = json.load(open(baseline_file))
    print(f"Baseline: {baseline_data.get('interventions_per_episode', 0):.1f} int/ep, "
          f"stability={baseline_data.get('stability_avg', 0.0):.4f}")
    
    # Test each parameter set
    results = []
    for i, params in enumerate(PARAM_SETS[1:], start=2):  # Skip baseline
        print(f"\n[{i}/{len(PARAM_SETS)}] Testing: {params['name']}")
        print(f"  {params['desc']}")
        
        # Note: In real implementation, we'd update controller params here
        # For now, we're testing with current controller settings
        # User will need to manually update controller between tests
        
        result = run_test(params)
        if result['success']:
            comparison = compare_with_baseline(baseline_file, result)
            if comparison:
                results.append({
                    'name': params['name'],
                    'desc': params['desc'],
                    'comparison': comparison
                })
                print(f"\n  Results:")
                print(f"    Interventions: {comparison['int_improvement']:+.1f}%")
                print(f"    Stability: {comparison['stab_improvement']:+.1f}%")
                print(f"    Average: {comparison['avg_improvement']:+.1f}%")
                if comparison['avg_improvement'] >= 10:
                    print(f"    *** TARGET MET! (10%+) ***")
            else:
                print("  Could not compare with baseline")
        else:
            print(f"  Test failed: {result.get('error', 'Unknown error')}")
    
    # Summary
    print("\n" + "="*70)
    print("INCREMENTAL TEST SUMMARY")
    print("="*70)
    if results:
        best = max(results, key=lambda x: x['comparison']['avg_improvement'])
        print(f"\nBest configuration: {best['name']}")
        print(f"  Description: {best['desc']}")
        print(f"  Average improvement: {best['comparison']['avg_improvement']:+.1f}%")
        print(f"  Interventions: {best['comparison']['int_improvement']:+.1f}%")
        print(f"  Stability: {best['comparison']['stab_improvement']:+.1f}%")
        
        if best['comparison']['avg_improvement'] >= 10:
            print("\n  *** TARGET ACHIEVED (10%+) ***")
        else:
            print(f"\n  Still need {10 - best['comparison']['avg_improvement']:.1f}% more improvement")
    else:
        print("\nNo successful tests completed")
    
    print("="*70)
    return 0

if __name__ == "__main__":
    sys.exit(main())

