#!/usr/bin/env python3
"""
Compare EDON v6.1 vs Baseline across multiple seeds.

Loads JSON results for:
- results/baseline_highstress_seed_{0-4}.json
- results/edon_v61_highstress_seed_{0-4}.json

Computes mean interventions and stability, then prints deltas.
"""

import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple


def load_results(prefix: str, num_seeds: int = 5) -> Tuple[float, float]:
    """
    Load results from multiple seed files and compute means.
    
    Args:
        prefix: File prefix (e.g., "baseline_highstress_seed" or "edon_v61_highstress_seed")
        num_seeds: Number of seed files to load (0 to num_seeds-1)
    
    Returns:
        (mean_interventions, mean_stability)
    """
    interventions = []
    stability = []
    
    for seed in range(num_seeds):
        # Handle both formats: prefix_seed_X.json or prefix_X.json
        if prefix.endswith("_seed"):
            file_path = Path(f"results/{prefix}_{seed}.json")
        else:
            file_path = Path(f"results/{prefix}_seed_{seed}.json")
        if not file_path.exists():
            print(f"[WARNING] File not found: {file_path}")
            continue
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Handle both formats: with run_metrics or direct
            if "run_metrics" in data:
                metrics = data["run_metrics"]
            else:
                metrics = data
            
            interventions.append(metrics.get("interventions_per_episode", 0.0))
            stability.append(metrics.get("stability_avg", 0.0))
        except Exception as e:
            print(f"[ERROR] Failed to load {file_path}: {e}")
            continue
    
    if len(interventions) == 0:
        raise ValueError(f"No valid results found for prefix: {prefix}")
    
    mean_int = sum(interventions) / len(interventions)
    mean_stab = sum(stability) / len(stability)
    
    return mean_int, mean_stab


def main():
    """Main comparison function."""
    print("="*70)
    print("EDON v6.1 vs Baseline Comparison")
    print("="*70)
    print()
    
    try:
        # Load baseline results
        base_int, base_stab = load_results("baseline_highstress_seed", num_seeds=5)
        print(f"Baseline: interventions={base_int:.2f}, stability={base_stab:.4f}")
        
        # Load EDON v6.1 results
        edon_int, edon_stab = load_results("edon_v61_highstress_seed", num_seeds=5)
        print(f"EDON v6.1: interventions={edon_int:.2f}, stability={edon_stab:.4f}")
        print()
        
        # Compute deltas
        if base_int > 0:
            delta_int = 100.0 * (base_int - edon_int) / base_int
        else:
            delta_int = 0.0
        
        if base_stab > 0:
            delta_stab = 100.0 * (base_stab - edon_stab) / base_stab
        else:
            delta_stab = 0.0
        
        avg_delta = (delta_int + delta_stab) / 2.0
        
        print(f"Delta Interventions%: {delta_int:+.2f}%")
        print(f"Delta Stability%: {delta_stab:+.2f}%")
        print(f"Average%: {avg_delta:+.2f}%")
        print("="*70)
        
    except Exception as e:
        print(f"[ERROR] Comparison failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

