# V4 Multi-Seed Validation

## Configuration (LOCKED)

```python
EDON_V31_HS_V4_CFG = {
    "BASE_GAIN": 0.5,
    "INSTABILITY_WEIGHT": 0.4,
    "DISTURBANCE_WEIGHT": 0.2,
    "RECOVERY_BOOST": 1.2,
    "PREFALL_MIN": 0.15,
    "PREFALL_RANGE": 0.50,
    "PREFALL_MAX": 0.70,
    "SAFE_THRESHOLD": 0.75,
    "SAFE_GAIN": 0.12,
}
```

## Validation Process

1. Run N_SEEDS = 5 different random seeds
2. For each seed:
   - Run 30-episode baseline
   - Run 30-episode EDON
   - Compute % improvements
3. Compute mean ± std across all seeds
4. Generate configuration label: "High_stress V4: ~X% ± Y% avg improvement"

## Usage

```bash
# Run validation for high_stress
python validate_v4_multi_seed.py

# Results saved to: results/v4_validation_high_stress_summary.json
```

## Expected Output

```
======================================================================
SUMMARY STATISTICS
======================================================================

Interventions:
  Mean: +X.X%
  Std:  X.X%
  Range: +X.X% to +X.X%

Stability:
  Mean: +X.X%
  Std:  X.X%
  Range: +X.X% to +X.X%

Average Improvement:
  Mean: +X.X%
  Std:  X.X%
  Range: +X.X% to +X.X%

======================================================================
CONFIGURATION LABEL
======================================================================

High stress V4: ~X.X% ± X.X% avg improvement
  (Based on 5 seeds, 30 episodes each)
```

## Next Steps

After high_stress validation:
1. Update script for medium_stress
2. Update script for light_stress  
3. Update script for hell_stress
4. Generate final configuration labels for all profiles

