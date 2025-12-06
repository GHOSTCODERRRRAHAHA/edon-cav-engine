"""
Helper script to run baseline vs v8 evaluation on hell_stress profile.

Runs both evaluations and prints commands for manual execution.
"""

import subprocess
import sys
from pathlib import Path


def main():
    print("="*80)
    print("EDON v8 Evaluation Helper - hell_stress Profile")
    print("="*80)
    print()
    
    # Commands to run
    baseline_cmd = [
        "python", "run_eval.py",
        "--mode", "baseline",
        "--profile", "hell_stress",
        "--episodes", "30",
        "--seed", "42",
        "--output", "results/baseline_v8_hell.json",
        "--edon-score"
    ]
    
    v8_cmd = [
        "python", "run_eval.py",
        "--mode", "edon",
        "--profile", "hell_stress",
        "--episodes", "30",
        "--seed", "42",
        "--output", "results/edon_v8_strategy_hell.json",
        "--edon-gain", "1.0",
        "--edon-arch", "v8_strategy",
        "--edon-score"
    ]
    
    print("Commands to run:")
    print()
    print("Baseline:")
    print("  " + " ".join(baseline_cmd))
    print()
    print("EDON v8:")
    print("  " + " ".join(v8_cmd))
    print()
    print("="*80)
    print()
    
    # Ask user if they want to run automatically
    response = input("Run evaluations automatically? (y/n): ").strip().lower()
    
    if response == 'y':
        print("\nRunning baseline evaluation...")
        print("-"*80)
        result1 = subprocess.run(baseline_cmd)
        
        if result1.returncode == 0:
            print("\nBaseline evaluation complete!")
        else:
            print("\nBaseline evaluation failed!")
            return
        
        print("\nRunning v8 evaluation...")
        print("-"*80)
        result2 = subprocess.run(v8_cmd)
        
        if result2.returncode == 0:
            print("\nv8 evaluation complete!")
            print("\nResults saved to:")
            print("  - results/baseline_v8_hell.json")
            print("  - results/edon_v8_strategy_hell.json")
            print("\nRun comparison:")
            print("  python training/compare_v8_vs_baseline.py \\")
            print("    --baseline results/baseline_v8_hell.json \\")
            print("    --v8 results/edon_v8_strategy_hell.json")
        else:
            print("\nv8 evaluation failed!")
    else:
        print("\nCopy and paste the commands above to run manually.")


if __name__ == "__main__":
    main()

