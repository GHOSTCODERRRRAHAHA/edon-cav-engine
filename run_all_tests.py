#!/usr/bin/env python3
"""
Comprehensive evaluation script for EDON controller across multiple scenarios.
Runs baseline and EDON tests for normal_stress, high_stress, and hell_stress profiles.
"""

import subprocess
import sys
import os
from pathlib import Path

# Ensure results directory exists
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Test configuration
STRESS_PROFILES = ["normal_stress", "high_stress", "hell_stress"]
EDON_GAINS = [0.60, 0.75, 0.90, 1.00]
EPISODES = 30

def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"\n❌ ERROR: {description} failed with exit code {result.returncode}")
        return False
    else:
        print(f"\n✅ SUCCESS: {description} completed")
        return True

def main():
    print("="*70)
    print("COMPREHENSIVE EDON EVALUATION TEST SUITE")
    print("="*70)
    print(f"Profiles: {', '.join(STRESS_PROFILES)}")
    print(f"EDON Gains: {', '.join(map(str, EDON_GAINS))}")
    print(f"Episodes per test: {EPISODES}")
    print("="*70)
    
    all_passed = True
    
    # Step 1: Run baseline tests for all profiles
    print("\n" + "="*70)
    print("STEP 1: BASELINE TESTS (No EDON)")
    print("="*70)
    
    for profile in STRESS_PROFILES:
        output_file = f"results/baseline_{profile}_v44.json"
        cmd = [
            sys.executable, "run_eval.py",
            "--mode", "baseline",
            "--episodes", str(EPISODES),
            "--profile", profile,
            "--output", output_file
        ]
        if not run_command(cmd, f"Baseline test: {profile}"):
            all_passed = False
    
    # Step 2: Run EDON tests for all profiles and gains
    print("\n" + "="*70)
    print("STEP 2: EDON TESTS (Multiple Gains)")
    print("="*70)
    
    for profile in STRESS_PROFILES:
        for gain in EDON_GAINS:
            tag = f"{int(gain * 100):03d}"
            output_file = f"results/edon_{profile}_v44_g{tag}.json"
            cmd = [
                sys.executable, "run_eval.py",
                "--mode", "edon",
                "--episodes", str(EPISODES),
                "--profile", profile,
                "--edon-gain", str(gain),
                "--edon-controller-version", "v3",
                "--output", output_file
            ]
            if not run_command(cmd, f"EDON test: {profile}, gain={gain}"):
                all_passed = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUITE COMPLETE")
    print("="*70)
    if all_passed:
        print("✅ All tests completed successfully!")
        print("\nNext steps:")
        print("1. Run plot_results.py to visualize comparisons")
        print("2. Check results/ directory for JSON files")
        print("3. Compare baseline vs EDON metrics for each profile")
    else:
        print("❌ Some tests failed. Check output above for details.")
    print("="*70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
