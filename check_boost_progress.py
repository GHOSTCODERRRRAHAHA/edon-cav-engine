#!/usr/bin/env python3
"""Check progress of predicted boost incremental tests."""

import json
from pathlib import Path
import numpy as np

PROFILE = "medium_stress"
results = []

for test in range(1, 6):
    for seed in range(1, 4):
        bf = f'results/baseline_boost_test{test}_seed{seed}_{PROFILE}.json'
        ef = f'results/edon_boost_test{test}_seed{seed}_{PROFILE}.json'
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
    
    configs = {
        1: "Boost 0.15, Scale 1.5, Cap 0.5 (7.5% max)",
        2: "Boost 0.20, Scale 2.0, Cap 0.5 (10% max) - Current V5.2",
        3: "Boost 0.25, Scale 2.5, Cap 0.5 (12.5% max)",
        4: "Boost 0.20, Scale 1.5, Cap 0.5 (7.5% max)",
        5: "Boost 0.20, Scale 2.0, Cap 0.6 (12% max)"
    }
    
    print("Current Progress:")
    for t in sorted(tests.keys()):
        mean = np.mean(tests[t])
        std = np.std(tests[t]) if len(tests[t]) > 1 else 0.0
        status = "PASS" if 1.0 <= mean <= 3.0 else "CLOSE" if mean > 0 else "FAIL"
        print(f"  Test {t} ({configs[t]}): {mean:+.1f}% ± {std:.1f}% (n={len(tests[t])} seeds) [{status}]")
    
    if len(tests) == 5 and all(len(tests[t]) == 3 for t in tests):
        print("\nAll tests complete!")
        best = max(tests.items(), key=lambda x: np.mean(x[1]))
        print(f"Best: Test {best[0]} ({configs[best[0]]}) = {np.mean(best[1]):+.1f}%")
        if 1.0 <= np.mean(best[1]) <= 3.0:
            print("  ✅ TARGET MET!")
else:
    print("Tests still running or no results yet...")

