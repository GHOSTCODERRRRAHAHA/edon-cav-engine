#!/usr/bin/env python3
"""Validate V4 configuration with 100-episode tests to reduce variance."""

import json
from pathlib import Path

baseline_file = "results/baseline_high_stress_v31_v4_100ep.json"
edon_file = "results/edon_high_stress_v31_v4_100ep.json"

print("="*70)
print("EDON v3.1 High-Stress V4 - 100 Episode Validation")
print("="*70)

# Check if files exist
if not Path(baseline_file).exists():
    print(f"\n[WAITING] Baseline test not complete: {baseline_file}")
    print("  Run: python run_eval.py --mode baseline --episodes 100 --profile high_stress --output results/baseline_high_stress_v31_v4_100ep.json")
    baseline = None
else:
    baseline = json.load(open(baseline_file))
    print(f"\n[OK] Baseline loaded: {baseline_file}")

if not Path(edon_file).exists():
    print(f"\n[WAITING] EDON test not complete: {edon_file}")
    print("  Run: python run_eval.py --mode edon --episodes 100 --profile high_stress --edon-gain 0.75 --edon-controller-version v3 --output results/edon_high_stress_v31_v4_100ep.json")
    edon = None
else:
    edon = json.load(open(edon_file))
    print(f"\n[OK] EDON loaded: {edon_file}")

if baseline and edon:
    bi = baseline['interventions_per_episode']
    ei = edon['interventions_per_episode']
    bs = baseline['stability_avg']
    es = edon['stability_avg']
    
    int_imp = ((bi - ei) / bi) * 100 if bi > 0 else 0
    stab_imp = ((bs - es) / bs) * 100 if bs > 0 else 0
    avg_imp = (int_imp + stab_imp) / 2
    
    print("\n" + "="*70)
    print("VALIDATION RESULTS (100 episodes)")
    print("="*70)
    print(f"\nBaseline:")
    print(f"  Interventions: {bi:.1f}/ep")
    print(f"  Stability: {bs:.4f}")
    print(f"\nEDON v3.1 HS V4:")
    print(f"  Interventions: {ei:.1f}/ep")
    print(f"  Stability: {es:.4f}")
    print(f"\nImprovements:")
    print(f"  Interventions: {int_imp:+.1f}%")
    print(f"  Stability: {stab_imp:+.1f}%")
    print(f"  Average: {avg_imp:+.1f}%")
    print("\n" + "="*70)
    
    # Check if in Stage 1 range (5-15%)
    if 5.0 <= avg_imp <= 15.0:
        print("[PASS] STAGE 1 ACHIEVED (5-15% improvement)")
        print(f"   Configuration is stable and validated at {avg_imp:.1f}%")
    elif avg_imp >= 10.0:
        print("[PASS] TARGET MET (10%+ improvement)")
        print(f"   Configuration exceeds target at {avg_imp:.1f}%")
    elif avg_imp >= 4.0:
        print("[WARN] STAGE 0-1 BORDERLINE (4-5% improvement)")
        print(f"   Configuration is at {avg_imp:.1f}% - close to Stage 1")
    elif avg_imp >= 0.0:
        print("[FAIL] BELOW STAGE 1 (<5% improvement)")
        print(f"   Configuration at {avg_imp:.1f}% - needs improvement")
    else:
        print("[FAIL] NEGATIVE IMPROVEMENT")
        print(f"   Configuration at {avg_imp:.1f}% - worse than baseline")
        print("   This suggests variance or configuration issue")
    
    print("="*70)
else:
    print("\n[INFO] Waiting for tests to complete...")
    print("  Check status with: python check_test_status.py")

