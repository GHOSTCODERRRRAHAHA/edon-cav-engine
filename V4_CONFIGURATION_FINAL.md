# EDON v3.1 High-Stress V4 - Final Configuration (LOCKED)

## Status: FROZEN ✅ (Performance: NOT VALIDATED)

This configuration is **locked** and serves as the reference baseline.

**⚠️ WARNING**: Multi-seed validation shows **-1.7% ± 1.6%** average improvement (worse than baseline). See `V4_MULTI_SEED_RESULTS.md` for details.

## Configuration Parameters

```python
EDON_V31_HS_V4_CFG = {
    "BASE_GAIN": 0.5,  # Updated from 0.4 (incremental testing showed +7.0% with 0.5)
    "INSTABILITY_WEIGHT": 0.4,
    "DISTURBANCE_WEIGHT": 0.2,
    "RECOVERY_BOOST": 1.2,  # 20% extra during recovery
    "PREFALL_MIN": 0.15,
    "PREFALL_RANGE": 0.50,  # Updated from 0.45 (incremental testing showed +4.7% with 0.50)
    "PREFALL_MAX": 0.70,  # Updated from 0.65 (to match range)
    "SAFE_THRESHOLD": 0.75,
    "SAFE_GAIN": 0.12,
}
```

## Multi-Seed Validation

### Process
1. Run **N_SEEDS = 5** different random seeds
2. For each seed:
   - Run 30-episode baseline
   - Run 30-episode EDON
   - Compute % improvements
3. Compute **mean ± std** across all seeds
4. Generate configuration label: **"High_stress V4: ~X% ± Y% avg improvement"**

### Scripts
- **`validate_v4_multi_seed.py`**: Single profile validation (high_stress)
- **`validate_v4_all_profiles.py`**: All profiles validation (light, medium, high, hell)

### Usage

```bash
# Validate high_stress only
python validate_v4_multi_seed.py

# Validate all profiles
python validate_v4_all_profiles.py
```

### Expected Output Format

```
High stress V4: ~X.X% ± X.X% avg improvement
  (Based on 5 seeds, 30 episodes each)
```

## Incremental Testing Results

| Test | Config | Average | Status |
|------|--------|---------|--------|
| Test 1 | BASE_GAIN=0.5 | +7.0% | ✅ Best single run |
| Test 8 | BASE_GAIN=0.5, PREFALL_RANGE=0.50 | +4.7% | ⚠️ Most consistent |
| Final | BASE_GAIN=0.5, PREFALL_RANGE=0.50 | +2.9% | ⚠️ Latest run |

**Note**: High variance in 30-episode tests (±3-4% swings). Multi-seed validation will provide stable band estimate.

## Files

- **Config**: `evaluation/edon_controller_v3.py` (EDON_V31_HS_V4_CFG)
- **Single profile validator**: `validate_v4_multi_seed.py`
- **All profiles validator**: `validate_v4_all_profiles.py`
- **Documentation**: `V4_MULTI_SEED_VALIDATION.md`

## Next Steps

1. ⏳ Run multi-seed validation for high_stress
2. ⏳ Run multi-seed validation for medium_stress
3. ⏳ Run multi-seed validation for light_stress
4. ⏳ Run multi-seed validation for hell_stress
5. Generate final configuration labels for all profiles

## Goal

**Not a perfect number, but a stable band** - the multi-seed validation will show the mean ± std improvement range, giving us confidence in the configuration's performance.

