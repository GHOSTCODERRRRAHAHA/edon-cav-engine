#!/usr/bin/env python3
"""Check progress of LPF incremental tests."""

import json
from pathlib import Path
import numpy as np

results = []
for test in range(1, 6):
    for seed in range(1, 4):
        bf = f'results/baseline_lpf_test{test}_seed{seed}.json'
        ef = f'results/edon_lpf_test{test}_seed{seed}.json'
        if Path(bf).exists() and Path(ef).exists():
            b = json.load(open(bf))
            e = json.load(open(ef))
            bi = b['interventions_per_episode']
            ei = e['interventions_per_episode']
            bs = b['stability_avg']
            es = e['stability_avg']
            int_imp = ((bi - ei) / bi) * 100
            stab_imp = ((bs - es) / bs) * 100
            avg_imp = (int_imp + stab_imp) / 2
            results.append((test, seed, avg_imp))

if results:
    tests = {}
    for t, s, a in results:
        if t not in tests:
            tests[t] = []
        tests[t].append(a)
    
    formulas = {
        1: "0.75 - 0.15 * instability",
        2: "0.80 - 0.20 * instability",
        3: "0.70 - 0.10 * instability",
        4: "0.85 - 0.25 * instability",
        5: "0.72 - 0.12 * instability"
    }
    
    print("Current Progress:")
    for t in sorted(tests.keys()):
        mean = np.mean(tests[t])
        std = np.std(tests[t]) if len(tests[t]) > 1 else 0.0
        print(f"  Test {t} ({formulas[t]}): {mean:+.1f}% ± {std:.1f}% (n={len(tests[t])} seeds)")
    
    if len(tests) == 5 and all(len(tests[t]) == 3 for t in tests):
        print("\nAll tests complete!")
        best = max(tests.items(), key=lambda x: np.mean(x[1]))
        print(f"Best: Test {best[0]} ({formulas[best[0]]}) = {np.mean(best[1]):+.1f}%")
else:
    print("Tests still running or no results yet...")

