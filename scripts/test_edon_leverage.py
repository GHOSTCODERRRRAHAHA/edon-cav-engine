"""
EDON Leverage Test

Tests whether EDON can actually move interventions at all by running
evaluations with different EDON configurations:
1. Baseline only (EDON OFF)
2. EDON normal gain (current)
3. EDON high gain (2x)
4. EDON inverted (flip sign)

This proves if EDON has leverage before continuing RL training.
"""

import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.compare_v8_vs_baseline import load_results, extract_metrics


def run_evaluation(config_name: str, edon_gain: float = None, edon_inverted: bool = False, 
                   episodes: int = 30, seed: int = 42) -> Dict[str, Any]:
    """Run evaluation with specific EDON configuration."""
    output_file = f"results/leverage_test_{config_name}.json"
    
    cmd = [
        "python", "run_eval.py",
        "--mode", "edon" if edon_gain is not None else "baseline",
        "--profile", "high_stress",
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--output", output_file,
        "--edon-score"
    ]
    
    if edon_gain is not None:
        # For v8, we need to modify the environment to apply gain
        # We'll use a custom script or modify run_eval.py temporarily
        # For now, use edon_gain parameter (though v8 doesn't use it directly)
        cmd.extend(["--edon-gain", str(edon_gain)])
        cmd.extend(["--edon-arch", "v8_strategy"])
    
    print(f"\n[{config_name}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return None
    
    # Load results
    if Path(output_file).exists():
        return load_results(output_file)
    return None


def test_edon_leverage():
    """Run leverage test with different EDON configurations."""
    print("="*80)
    print("EDON Leverage Test")
    print("="*80)
    print("\nThis test checks if EDON can actually move interventions.")
    print("If all configs give ~40 interventions, EDON has no leverage.")
    print("If high-gain or inverted changes interventions significantly, EDON has leverage.")
    print()
    
    results = {}
    
    # Test 1: Baseline only (EDON OFF)
    print("\n[1/4] Baseline Only (EDON OFF)")
    print("-"*80)
    baseline_results = run_evaluation("baseline", edon_gain=None)
    if baseline_results:
        baseline_metrics = extract_metrics(baseline_results)
        results["baseline"] = baseline_metrics
        print(f"  Interventions/ep: {baseline_metrics.get('interventions_per_episode', 'N/A')}")
        print(f"  Stability: {baseline_metrics.get('stability', 'N/A')}")
    
    # Test 2: EDON normal gain (current setup, gain=1.0)
    print("\n[2/4] EDON Normal Gain (gain=1.0)")
    print("-"*80)
    normal_results = run_evaluation("edon_normal", edon_gain=1.0)
    if normal_results:
        normal_metrics = extract_metrics(normal_results)
        results["edon_normal"] = normal_metrics
        print(f"  Interventions/ep: {normal_metrics.get('interventions_per_episode', 'N/A')}")
        print(f"  Stability: {normal_metrics.get('stability', 'N/A')}")
    
    # Test 3: EDON high gain (2x corrections)
    print("\n[3/4] EDON High Gain (gain=2.0)")
    print("-"*80)
    high_results = run_evaluation("edon_high", edon_gain=2.0)
    if high_results:
        high_metrics = extract_metrics(high_results)
        results["edon_high"] = high_metrics
        print(f"  Interventions/ep: {high_metrics.get('interventions_per_episode', 'N/A')}")
        print(f"  Stability: {high_metrics.get('stability', 'N/A')}")
    
    # Test 4: EDON inverted (negative gain as proxy)
    print("\n[4/4] EDON Inverted (gain=-1.0, doing opposite)")
    print("-"*80)
    inverted_results = run_evaluation("edon_inverted", edon_gain=-1.0)
    if inverted_results:
        inverted_metrics = extract_metrics(inverted_results)
        results["edon_inverted"] = inverted_metrics
        print(f"  Interventions/ep: {inverted_metrics.get('interventions_per_episode', 'N/A')}")
        print(f"  Stability: {inverted_metrics.get('stability', 'N/A')}")
    
    # Summary
    print("\n" + "="*80)
    print("Leverage Test Summary")
    print("="*80)
    
    if not results:
        print("ERROR: No results collected")
        return
    
    baseline_int = results.get("baseline", {}).get("interventions_per_episode", 0)
    
    print(f"\n{'Config':<20} {'Interventions/ep':<20} {'Stability':<15} {'Delta%':<15}")
    print("-"*80)
    
    for config_name, metrics in results.items():
        int_val = metrics.get("interventions_per_episode", 0)
        stab_val = metrics.get("stability", 0)
        
        if baseline_int > 0:
            delta_pct = ((int_val - baseline_int) / baseline_int) * 100
        else:
            delta_pct = 0.0
        
        print(f"{config_name:<20} {int_val:<20.2f} {stab_val:<15.4f} {delta_pct:>+6.1f}%")
    
    # Analysis
    print("\n" + "="*80)
    print("Analysis")
    print("="*80)
    
    if baseline_int > 0:
        normal_int = results.get("edon_normal", {}).get("interventions_per_episode", baseline_int)
        high_int = results.get("edon_high", {}).get("interventions_per_episode", baseline_int)
        inverted_int = results.get("edon_inverted", {}).get("interventions_per_episode", baseline_int)
        
        normal_delta = abs((normal_int - baseline_int) / baseline_int * 100)
        high_delta = abs((high_int - baseline_int) / baseline_int * 100)
        inverted_delta = abs((inverted_int - baseline_int) / baseline_int * 100)
        
        max_delta = max(normal_delta, high_delta, inverted_delta)
        
        if max_delta < 2.0:
            print("\n[CONCLUSION] EDON has WEAK leverage (<2% change)")
            print("  - Interventions barely change with EDON on/off/high/inverted")
            print("  - EDON's hook into controller doesn't affect intervention triggers")
            print("\n[RECOMMENDATION] Change where EDON acts:")
            print("  - Move EDON up a level (modulate gains/modes, not direct torques)")
            print("  - Use EDON as mode selector (normal/cautious/recovery)")
            print("  - Or intervention logic is based on things EDON can't affect")
        elif max_delta >= 10.0:
            print(f"\n[CONCLUSION] EDON has STRONG leverage ({max_delta:.1f}% change)")
            print("  - Interventions change significantly with EDON configuration")
            print("  - Control hook is fine, but policy needs better features/context")
            print("\n[RECOMMENDATION] Fix what EDON sees:")
            print("  - Add early-warning features (rolling variance, danger score)")
            print("  - Give EDON memory (GRU/RNN or stacked observations)")
            print("  - Policy needs temporal context to learn intervention causes")
        else:
            print(f"\n[CONCLUSION] EDON has MODERATE leverage ({max_delta:.1f}% change)")
            print("  - Some effect, but may need stronger signal or better features")
            print("\n[RECOMMENDATION] Try both:")
            print("  - Increase EDON gain/authority")
            print("  - Add temporal features and memory")
    
    print("="*80)


if __name__ == "__main__":
    test_edon_leverage()

