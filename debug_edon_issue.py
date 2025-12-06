#!/usr/bin/env python3
"""Debug script to understand why EDON is underperforming."""

import json
from pathlib import Path

# Load baseline and EDON results
baseline = json.load(open("results/baseline_high_stress_v44.json"))
edon = json.load(open("results/edon_high_v44_g075.json"))

print("="*70)
print("EDON UNDERPERFORMANCE ANALYSIS")
print("="*70)

print("\nBASELINE METRICS:")
print(f"  Interventions: {baseline['interventions_per_episode']:.1f}/ep")
print(f"  Stability: {baseline['stability_avg']:.4f}")
print(f"  Avg episode length: {baseline['avg_episode_length']:.1f} steps")

print("\nEDON (gain=0.75) METRICS:")
print(f"  Interventions: {edon['interventions_per_episode']:.1f}/ep")
print(f"  Stability: {edon['stability_avg']:.4f}")
print(f"  Avg episode length: {edon['avg_episode_length']:.1f} steps")

# Compute differences
int_diff = edon['interventions_per_episode'] - baseline['interventions_per_episode']
stab_diff = edon['stability_avg'] - baseline['stability_avg']
int_pct = (int_diff / baseline['interventions_per_episode']) * 100
stab_pct = (stab_diff / baseline['stability_avg']) * 100

print("\nDIFFERENCES:")
print(f"  Interventions: {int_diff:+.1f} ({int_pct:+.1f}%)")
print(f"  Stability: {stab_diff:+.4f} ({stab_pct:+.1f}%)")

# Check if EDON is making episodes shorter (more failures)
if edon['avg_episode_length'] < baseline['avg_episode_length']:
    print(f"\n[WARNING] EDON episodes are SHORTER ({edon['avg_episode_length']:.1f} vs {baseline['avg_episode_length']:.1f} steps)")
    print("   This suggests EDON is causing MORE failures (earlier interventions)")
else:
    print(f"\n✓ EDON episodes are LONGER ({edon['avg_episode_length']:.1f} vs {baseline['avg_episode_length']:.1f} steps)")

# Check stability components
print("\nSTABILITY BREAKDOWN:")
print(f"  Baseline roll RMS: {baseline.get('roll_rms_avg', 'N/A')}")
print(f"  EDON roll RMS: {edon.get('roll_rms_avg', 'N/A')}")
print(f"  Baseline pitch RMS: {baseline.get('pitch_rms_avg', 'N/A')}")
print(f"  EDON pitch RMS: {edon.get('pitch_rms_avg', 'N/A')}")

# Check prefall events
print("\nPREFALL EVENTS:")
print(f"  Baseline: {baseline.get('prefall_events_per_episode', 'N/A'):.1f}/ep")
print(f"  EDON: {edon.get('prefall_events_per_episode', 'N/A'):.1f}/ep")

print("\n" + "="*70)
print("HYPOTHESIS:")
print("="*70)
print("EDON corrections may be:")
print("1. Too weak (not enough correction magnitude)")
print("2. Wrong direction (destabilizing instead of stabilizing)")
print("3. Too aggressive (causing oscillations)")
print("4. Applied at wrong times (interfering with stable gait)")
print("="*70)

