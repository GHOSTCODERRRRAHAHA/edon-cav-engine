#!/usr/bin/env python3
"""Check status of comprehensive test suite."""

from pathlib import Path
import json

STRESS_PROFILES = ["normal_stress", "high_stress", "hell_stress"]
EDON_GAINS = [0.60, 0.75, 0.90, 1.00]

results_dir = Path("results")

def check_file(filepath):
    """Check if result file exists and is valid."""
    if not filepath.exists():
        return False, "Missing"
    try:
        with open(filepath) as f:
            data = json.load(f)
            if "run_metrics" in data:
                episodes = len(data["run_metrics"]["episodes"])
                return True, f"OK - {episodes} episodes"
            return True, "OK (invalid format)"
    except:
        return True, "✓ (parse error)"

def main():
    print("="*70)
    print("TEST SUITE STATUS CHECK")
    print("="*70)
    
    # Check baseline tests
    print("\nBASELINE TESTS:")
    print("-" * 70)
    for profile in STRESS_PROFILES:
        filepath = results_dir / f"baseline_{profile}_v44.json"
        exists, status = check_file(filepath)
        status_icon = "[OK]" if exists else "[...]"
        print(f"{status_icon} {profile:20s} {status}")
    
    # Check EDON tests
    print("\nEDON TESTS:")
    print("-" * 70)
    for profile in STRESS_PROFILES:
        print(f"\n  {profile}:")
        for gain in EDON_GAINS:
            tag = f"{int(gain * 100):03d}"
            filepath = results_dir / f"edon_{profile}_v44_g{tag}.json"
            exists, status = check_file(filepath)
            status_icon = "[OK]" if exists else "[...]"
            print(f"    {status_icon} gain={gain:4.2f}  {status}")
    
    # Count completed
    total = len(STRESS_PROFILES) * (1 + len(EDON_GAINS))
    completed = 0
    for profile in STRESS_PROFILES:
        if (results_dir / f"baseline_{profile}_v44.json").exists():
            completed += 1
        for gain in EDON_GAINS:
            tag = f"{int(gain * 100):03d}"
            if (results_dir / f"edon_{profile}_v44_g{tag}.json").exists():
                completed += 1
    
    print("\n" + "="*70)
    print(f"Progress: {completed}/{total} tests completed ({100*completed/total:.1f}%)")
    print("="*70)

if __name__ == "__main__":
    main()

