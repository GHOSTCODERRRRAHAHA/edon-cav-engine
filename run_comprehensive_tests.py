#!/usr/bin/env python3
"""
Comprehensive evaluation script for EDON controller.
Runs baseline and EDON tests across all stress profiles and gains.
"""

import subprocess
import sys
import time
from pathlib import Path

# Test configuration
STRESS_PROFILES = ["normal_stress", "high_stress", "hell_stress"]
EDON_GAINS = [0.60, 0.75, 0.90, 1.00]
EPISODES = 30

# Ensure results directory exists
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

def run_test(cmd, test_name):
    """Run a single test and return success status."""
    print(f"\n{'='*70}")
    print(f"Running: {test_name}")
    print(f"{'='*70}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        elapsed = time.time() - start_time
        print(f"\n✅ {test_name} completed in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {test_name} FAILED after {elapsed:.1f}s (exit code {e.returncode})")
        return False

def main():
    print("="*70)
    print("COMPREHENSIVE EDON EVALUATION TEST SUITE")
    print("="*70)
    print(f"Profiles: {', '.join(STRESS_PROFILES)}")
    print(f"EDON Gains: {', '.join(map(str, EDON_GAINS))}")
    print(f"Episodes per test: {EPISODES}")
    print(f"\nTotal tests: {len(STRESS_PROFILES) * (1 + len(EDON_GAINS))}")
    print("  - Baseline tests: 3")
    print(f"  - EDON tests: {len(STRESS_PROFILES) * len(EDON_GAINS)}")
    print("="*70)
    
    all_results = []
    test_num = 0
    total_tests = len(STRESS_PROFILES) * (1 + len(EDON_GAINS))
    
    # Step 1: Baseline tests
    print("\n" + "="*70)
    print("STEP 1: BASELINE TESTS (No EDON)")
    print("="*70)
    
    for profile in STRESS_PROFILES:
        test_num += 1
        output_file = f"results/baseline_{profile}_v44.json"
        cmd = [
            sys.executable, "run_eval.py",
            "--mode", "baseline",
            "--episodes", str(EPISODES),
            "--profile", profile,
            "--output", output_file
        ]
        success = run_test(cmd, f"[{test_num}/{total_tests}] Baseline: {profile}")
        all_results.append(("baseline", profile, None, success))
    
    # Step 2: EDON tests
    print("\n" + "="*70)
    print("STEP 2: EDON TESTS (Multiple Gains)")
    print("="*70)
    
    for profile in STRESS_PROFILES:
        for gain in EDON_GAINS:
            test_num += 1
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
            success = run_test(cmd, f"[{test_num}/{total_tests}] EDON: {profile}, gain={gain}")
            all_results.append(("edon", profile, gain, success))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUITE SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, _, _, success in all_results if success)
    failed = total_tests - passed
    
    print(f"Total tests: {total_tests}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed > 0:
        print("\nFailed tests:")
        for mode, profile, gain, success in all_results:
            if not success:
                if gain is None:
                    print(f"  - {mode}: {profile}")
                else:
                    print(f"  - {mode}: {profile}, gain={gain}")
    
    print("\n" + "="*70)
    if failed == 0:
        print("✅ ALL TESTS PASSED!")
        print("\nNext steps:")
        print("1. Check results/ directory for JSON files")
        print("2. Run plot_results.py to visualize comparisons")
        print("3. Compare baseline vs EDON metrics for each profile")
    else:
        print("❌ SOME TESTS FAILED")
        print("Check output above for error details")
    print("="*70)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

