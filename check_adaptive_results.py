#!/usr/bin/env python3
"""Check adaptive gain test results with different EDON gains."""

import json
from pathlib import Path

baseline = json.load(open("results/test_adaptive_baseline.json"))
bi = baseline['interventions_per_episode']
bs = baseline['stability_avg']

print("V4 Configuration with Different EDON Gains (30 episodes):")
print(f"Baseline: {bi:.1f} int/ep, stability={bs:.4f}")
print()

gains = [0.60, 0.75, 0.90]
results = []

for g in gains:
    if g == 0.75:
        fname = "results/test_adaptive_v4_30ep.json"
    else:
        fname = f"results/test_adaptive_v4_g{int(g*100):03d}.json"
    
    if Path(fname).exists():
        edon = json.load(open(fname))
        ei = edon['interventions_per_episode']
        es = edon['stability_avg']
        
        int_imp = ((bi - ei) / bi) * 100
        stab_imp = ((bs - es) / bs) * 100
        avg_imp = (int_imp + stab_imp) / 2
        
        results.append((g, int_imp, stab_imp, avg_imp))
        status = "PASS" if avg_imp >= 10 else "NEED MORE"
        print(f"Gain {g:.2f}: {int_imp:+.1f}% int, {stab_imp:+.1f}% stab, {avg_imp:+.1f}% avg [{status}]")
    else:
        print(f"Gain {g:.2f}: [Not tested yet]")

if results:
    best = max(results, key=lambda x: x[3])
    print()
    print(f"Best gain: {best[0]:.2f} ({best[3]:+.1f}% average)")

