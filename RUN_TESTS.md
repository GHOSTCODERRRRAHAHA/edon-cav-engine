# Comprehensive Test Suite - Running Instructions

## Overview

This test suite runs baseline and EDON evaluations across:
- **3 Stress Profiles**: normal_stress, high_stress, hell_stress
- **4 EDON Gains**: 0.60, 0.75, 0.90, 1.00
- **30 Episodes** per test

**Total: 15 tests** (3 baseline + 12 EDON)

## Quick Start

### Option 1: Run All Tests (Python)
```bash
python run_comprehensive_tests.py
```

### Option 2: Run All Tests (PowerShell)
```powershell
.\run_edon_tests.ps1
```

### Option 3: Run Manually (PowerShell)
```powershell
$profiles = @("normal_stress", "high_stress", "hell_stress")
$gains = @(0.60, 0.75, 0.90, 1.00)

# Baseline tests
foreach ($p in $profiles) {
    python run_eval.py --mode baseline --episodes 30 --profile $p --output "results/baseline_${p}_v44.json"
}

# EDON tests
foreach ($p in $profiles) {
    foreach ($g in $gains) {
        $tag = "{0:000}" -f [int]($g * 100)
        python run_eval.py --mode edon --episodes 30 --profile $p --edon-gain $g --edon-controller-version v3 --output "results/edon_${p}_v44_g${tag}.json"
    }
}
```

## Check Progress

```bash
python check_test_status.py
```

## Analyze Results

After all tests complete:

```bash
python analyze_results.py
```

This will show:
- Baseline metrics for each profile
- EDON metrics for each gain
- Percentage improvements (target: 5-15%+)
- Best gain per profile

## Expected Output Files

### Baseline:
- `results/baseline_normal_stress_v44.json`
- `results/baseline_high_stress_v44.json`
- `results/baseline_hell_stress_v44.json`

### EDON:
- `results/edon_normal_stress_v44_g060.json`
- `results/edon_normal_stress_v44_g075.json`
- `results/edon_normal_stress_v44_g090.json`
- `results/edon_normal_stress_v44_g100.json`
- (Same pattern for high_stress and hell_stress)

## Time Estimate

- Each test: ~30-60 seconds (30 episodes)
- Total time: ~15-30 minutes for all tests

## Target Metrics

We're aiming for **5-15%+ improvements** in:
- **Interventions**: Fewer operator interventions
- **Stability**: Lower stability score (better stability)
- **Freezes**: Fewer freeze/hesitation events

