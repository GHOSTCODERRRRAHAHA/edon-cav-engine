#!/usr/bin/env python3
"""Quick analysis of existing test results."""

import json
from pathlib import Path

def analyze_profile(profile_name, baseline_file, edon_files):
    """Analyze a single profile."""
    print(f"\n{'='*70}")
    print(f"{profile_name.upper()}")
    print(f"{'='*70}")
    
    # Load baseline
    baseline_path = Path(f"results/{baseline_file}")
    if not baseline_path.exists():
        print(f"Baseline not found: {baseline_file}")
        return
    
    baseline = json.load(open(baseline_path))
    base_int = baseline['interventions_per_episode']
    base_stab = baseline['stability_avg']
    
    print(f"Baseline: {base_int:.1f} interventions/ep, stability={base_stab:.4f}")
    print()
    
    best_gain = None
    best_avg = -999
    
    for gain, edon_file in edon_files:
        edon_path = Path(f"results/{edon_file}")
        if not edon_path.exists():
            print(f"Gain {gain:.2f}: [MISSING] {edon_file}")
            continue
        
        edon = json.load(open(edon_path))
        edon_int = edon['interventions_per_episode']
        edon_stab = edon['stability_avg']
        
        int_imp = ((base_int - edon_int) / base_int) * 100
        stab_imp = ((base_stab - edon_stab) / base_stab) * 100
        avg_imp = (int_imp + stab_imp) / 2
        
        int_status = "[GOOD]" if int_imp >= 5 else "[OK]" if int_imp >= 0 else "[BAD]"
        stab_status = "[GOOD]" if stab_imp >= 5 else "[OK]" if stab_imp >= 0 else "[BAD]"
        avg_status = "[GOOD]" if avg_imp >= 5 else "[OK]" if avg_imp >= 0 else "[BAD]"
        
        print(f"Gain {gain:.2f}:")
        print(f"  Interventions: {edon_int:.1f}/ep  {int_status} {int_imp:+.1f}%")
        print(f"  Stability:     {edon_stab:.4f}      {stab_status} {stab_imp:+.1f}%")
        print(f"  Average:       {avg_status} {avg_imp:+.1f}%")
        print()
        
        if avg_imp > best_avg:
            best_avg = avg_imp
            best_gain = gain
    
    if best_gain is not None:
        print(f"[BEST] Best gain: {best_gain:.2f} ({best_avg:+.1f}% average improvement)")

def main():
    print("="*70)
    print("EDON PERFORMANCE ANALYSIS - QUICK CHECK")
    print("="*70)
    
    # High stress
    analyze_profile(
        "HIGH_STRESS",
        "baseline_high_stress_v44.json",
        [
            (0.60, "edon_high_v44_g060.json"),
            (0.75, "edon_high_v44_g075.json"),
            (0.90, "edon_high_v44_g090.json"),
            (1.00, "edon_high_v44_g100.json"),
        ]
    )
    
    # Hell stress
    analyze_profile(
        "HELL_STRESS",
        "baseline_hell_stress_v44.json",
        [
            (0.60, "edon_hell_v44_g060.json"),
            (0.75, "edon_hell_v44_g075.json"),
            (0.90, "edon_hell_v44_g090.json"),
            (1.00, "edon_hell_v44_g100.json"),
        ]
    )
    
    # Normal stress (if available)
    if Path("results/baseline_normal_stress_v44.json").exists():
        analyze_profile(
            "NORMAL_STRESS",
            "baseline_normal_stress_v44.json",
            [
                (0.60, "edon_normal_stress_v44_g060.json"),
                (0.75, "edon_normal_stress_v44_g075.json"),
                (0.90, "edon_normal_stress_v44_g090.json"),
                (1.00, "edon_normal_stress_v44_g100.json"),
            ]
        )
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("Target: 5-15%+ improvements")
    print("[GOOD] = >=5% improvement")
    print("[OK] = 0-5% improvement")
    print("[BAD] = negative (worse than baseline)")
    print("="*70)

if __name__ == "__main__":
    main()

