#!/usr/bin/env python3
"""
Incremental testing for V4.1 Adaptive Gain System.

Tests different parameter combinations and saves the configuration
that achieves the highest percentage improvement consistently.
"""

import subprocess
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

# Test configurations to try incrementally
TEST_CONFIGS = [
    # Test 1: V4.1 baseline (BASE_GAIN=0.50, PREFALL=0.15-0.60)
    {
        "name": "V4.1_baseline",
        "base_gain": 0.50,
        "prefall_min": 0.15,
        "prefall_max": 0.60,
        "safe_threshold": 0.75,
        "safe_gain": 0.12,
    },
    # Test 2: Higher BASE_GAIN (0.55)
    {
        "name": "V4.1_base_055",
        "base_gain": 0.55,
        "prefall_min": 0.15,
        "prefall_max": 0.60,
        "safe_threshold": 0.75,
        "safe_gain": 0.12,
    },
    # Test 3: Wider PREFALL range (0.10-0.65)
    {
        "name": "V4.1_prefall_wide",
        "base_gain": 0.50,
        "prefall_min": 0.10,
        "prefall_max": 0.65,
        "safe_threshold": 0.75,
        "safe_gain": 0.12,
    },
    # Test 4: Narrower PREFALL range (0.18-0.55)
    {
        "name": "V4.1_prefall_narrow",
        "base_gain": 0.50,
        "prefall_min": 0.18,
        "prefall_max": 0.55,
        "safe_threshold": 0.75,
        "safe_gain": 0.12,
    },
    # Test 5: Higher PREFALL max (0.15-0.70)
    {
        "name": "V4.1_prefall_high",
        "base_gain": 0.50,
        "prefall_min": 0.15,
        "prefall_max": 0.70,
        "safe_threshold": 0.75,
        "safe_gain": 0.12,
    },
    # Test 6: Lower BASE_GAIN (0.45) with wider PREFALL
    {
        "name": "V4.1_base_045_wide",
        "base_gain": 0.45,
        "prefall_min": 0.12,
        "prefall_max": 0.65,
        "safe_threshold": 0.75,
        "safe_gain": 0.12,
    },
]

# Test parameters
N_SEEDS = 3  # Number of random seeds per configuration
EPISODES_PER_SEED = 30
PROFILE = "high_stress"  # Start with high_stress (most challenging)
EDON_GAIN = 0.75  # Fixed EDON gain parameter

RESULTS_DIR = Path("results/v41_incremental")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def apply_config_to_code(config: Dict) -> None:
    """Apply test configuration to run_eval.py by modifying the code."""
    import re
    
    # Read current code with UTF-8 encoding
    with open("run_eval.py", "r", encoding="utf-8") as f:
        code = f.read()
    
    # Replace BASE_GAIN (handle Unicode arrow character)
    code = re.sub(
        r'BASE_GAIN = \d+\.\d+\s+#.*',
        f'BASE_GAIN = {config["base_gain"]:.2f}  # Was 0.40 -> better stability backbone',
        code
    )
    
    # Replace PREFALL_MIN (look for PREFALL_MIN = 0.15)
    code = re.sub(
        r'PREFALL_MIN = \d+\.\d+',
        f'PREFALL_MIN = {config["prefall_min"]:.2f}',
        code
    )
    
    # Replace PREFALL_MAX (look for PREFALL_MAX = 0.60)
    code = re.sub(
        r'PREFALL_MAX = \d+\.\d+',
        f'PREFALL_MAX = {config["prefall_max"]:.2f}',
        code
    )
    
    # Replace SAFE_THRESHOLD (look for catastrophic_risk > 0.75)
    code = re.sub(
        r'catastrophic_risk > \d+\.\d+',
        f'catastrophic_risk > {config["safe_threshold"]:.2f}',
        code
    )
    
    # Replace SAFE_GAIN (look for SAFE_GAIN = 0.12)
    code = re.sub(
        r'SAFE_GAIN = \d+\.\d+',
        f'SAFE_GAIN = {config["safe_gain"]:.2f}',
        code
    )
    
    # Write back with UTF-8 encoding
    with open("run_eval.py", "w", encoding="utf-8") as f:
        f.write(code)


def run_evaluation(seed: int, config_name: str, mode: str) -> str:
    """Run a single evaluation and return output file path."""
    output_file = RESULTS_DIR / f"{config_name}_{mode}_seed{seed}.json"
    
    cmd = [
        "python", "run_eval.py",
        "--mode", mode,
        "--episodes", str(EPISODES_PER_SEED),
        "--profile", PROFILE,
        "--seed", str(seed),
        "--output", str(output_file),
    ]
    
    if mode == "edon":
        cmd.extend(["--edon-gain", str(EDON_GAIN)])
    
    print(f"  Running {mode} (seed {seed})...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr}")
        return None
    
    return str(output_file)


def compute_improvement(baseline_file: str, edon_file: str) -> Dict[str, float]:
    """Compute improvement metrics from result files."""
    try:
        with open(baseline_file, "r") as f:
            baseline_data = json.load(f)
        with open(edon_file, "r") as f:
            edon_data = json.load(f)
        
        baseline_metrics = baseline_data.get("run_metrics", {})
        edon_metrics = edon_data.get("run_metrics", {})
        
        # Compute percentage improvements
        baseline_interventions = baseline_metrics.get("interventions", 0)
        edon_interventions = edon_metrics.get("interventions", 0)
        interventions_improvement = (
            ((baseline_interventions - edon_interventions) / baseline_interventions * 100)
            if baseline_interventions > 0 else 0.0
        )
        
        baseline_stability = baseline_metrics.get("stability_score", 1.0)
        edon_stability = edon_metrics.get("stability_score", 1.0)
        stability_improvement = (
            ((baseline_stability - edon_stability) / baseline_stability * 100)
            if baseline_stability > 0 else 0.0
        )
        
        # Average improvement
        average_improvement = (interventions_improvement + stability_improvement) / 2.0
        
        return {
            "interventions": interventions_improvement,
            "stability": stability_improvement,
            "average": average_improvement,
        }
    except Exception as e:
        print(f"    ERROR computing improvement: {e}")
        return {"interventions": 0.0, "stability": 0.0, "average": 0.0}


def test_configuration(config: Dict) -> Dict:
    """Test a single configuration across multiple seeds."""
    print(f"\n{'='*70}")
    print(f"Testing: {config['name']}")
    print(f"  BASE_GAIN={config['base_gain']:.2f}")
    print(f"  PREFALL={config['prefall_min']:.2f}-{config['prefall_max']:.2f}")
    print(f"  SAFE threshold={config['safe_threshold']:.2f}, gain={config['safe_gain']:.2f}")
    print(f"{'='*70}")
    
    # Apply configuration to code
    apply_config_to_code(config)
    
    results = []
    
    # Run tests for each seed
    for seed in range(1, N_SEEDS + 1):
        print(f"\nSeed {seed}/{N_SEEDS}:")
        
        # Run baseline
        baseline_file = run_evaluation(seed, config["name"], "baseline")
        if baseline_file is None:
            continue
        
        # Run EDON
        edon_file = run_evaluation(seed, config["name"], "edon")
        if edon_file is None:
            continue
        
        # Compute improvement
        improvement = compute_improvement(baseline_file, edon_file)
        results.append(improvement)
        
        print(f"    Interventions: {improvement['interventions']:+.1f}%")
        print(f"    Stability: {improvement['stability']:+.1f}%")
        print(f"    Average: {improvement['average']:+.1f}%")
    
    if not results:
        return {
            "config": config,
            "results": [],
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    
    # Compute statistics
    averages = [r["average"] for r in results]
    mean_avg = statistics.mean(averages)
    std_avg = statistics.stdev(averages) if len(averages) > 1 else 0.0
    min_avg = min(averages)
    max_avg = max(averages)
    
    print(f"\n{config['name']} Summary:")
    print(f"  Mean: {mean_avg:+.2f}%")
    print(f"  Std:  {std_avg:.2f}%")
    print(f"  Range: {min_avg:+.2f}% to {max_avg:+.2f}%")
    
    return {
        "config": config,
        "results": results,
        "mean": mean_avg,
        "std": std_avg,
        "min": min_avg,
        "max": max_avg,
    }


def main():
    """Run incremental tests and save best configuration."""
    print("="*70)
    print("V4.1 Adaptive Gain System - Incremental Testing")
    print("="*70)
    print(f"Testing {len(TEST_CONFIGS)} configurations")
    print(f"  Seeds per config: {N_SEEDS}")
    print(f"  Episodes per seed: {EPISODES_PER_SEED}")
    print(f"  Profile: {PROFILE}")
    print(f"  EDON gain: {EDON_GAIN}")
    print("="*70)
    
    all_results = []
    
    # Test each configuration
    for config in TEST_CONFIGS:
        result = test_configuration(config)
        all_results.append(result)
    
    # Find best configuration (highest mean with reasonable consistency)
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    
    # Sort by mean improvement
    all_results.sort(key=lambda x: x["mean"], reverse=True)
    
    print("\nRankings (by mean improvement):")
    for i, result in enumerate(all_results, 1):
        config = result["config"]
        print(f"\n{i}. {config['name']}: {result['mean']:+.2f}% ± {result['std']:.2f}%")
        print(f"   BASE_GAIN={config['base_gain']:.2f}, PREFALL={config['prefall_min']:.2f}-{config['prefall_max']:.2f}")
        print(f"   Range: {result['min']:+.2f}% to {result['max']:+.2f}%")
    
    # Best configuration
    best = all_results[0]
    print(f"\n{'='*70}")
    print(f"BEST CONFIGURATION: {best['config']['name']}")
    print(f"{'='*70}")
    print(f"Mean improvement: {best['mean']:+.2f}% ± {best['std']:.2f}%")
    print(f"Range: {best['min']:+.2f}% to {best['max']:+.2f}%")
    print(f"\nParameters:")
    print(f"  BASE_GAIN: {best['config']['base_gain']:.2f}")
    print(f"  PREFALL_MIN: {best['config']['prefall_min']:.2f}")
    print(f"  PREFALL_MAX: {best['config']['prefall_max']:.2f}")
    print(f"  SAFE_THRESHOLD: {best['config']['safe_threshold']:.2f}")
    print(f"  SAFE_GAIN: {best['config']['safe_gain']:.2f}")
    
    # Save best configuration
    best_config_file = RESULTS_DIR / "best_config_v41.json"
    with open(best_config_file, "w") as f:
        json.dump({
            "config": best["config"],
            "statistics": {
                "mean": best["mean"],
                "std": best["std"],
                "min": best["min"],
                "max": best["max"],
            },
            "all_results": all_results,
        }, f, indent=2)
    
    print(f"\nBest configuration saved to: {best_config_file}")
    
    # Apply best configuration to code
    print(f"\nApplying best configuration to run_eval.py...")
    apply_config_to_code(best["config"])
    print("Done!")
    
    return best


if __name__ == "__main__":
    main()

