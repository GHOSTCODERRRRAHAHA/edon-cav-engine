# EDON v3.1 High-Stress V4 Configuration - LOCKED

## Status: FROZEN ✅

This configuration is **locked** and serves as the reference baseline. All parameters are defined in `EDON_V31_HS_V4_CFG` dictionary.

## Configuration Name
**EDON_v3.1_high_stress_V4**

## Performance (30 episodes, high_stress)
- **Interventions**: +4.4% improvement
- **Stability**: +8.4% improvement  
- **Average**: **+6.4%** improvement
- **EDON Gain**: 0.75 (optimal)

## Validation Status
- ⏳ **100-episode baseline**: [Running in background]
- ⏳ **100-episode EDON V4**: [Running in background]
- **Purpose**: Confirm improvements are stable (4-9% band) and not variance

## Locked Parameters

All parameters are in `EDON_V31_HS_V4_CFG` in `evaluation/edon_controller_v3.py`:

```python
EDON_V31_HS_V4_CFG = {
    # Adaptive gain settings
    "BASE_GAIN": 0.4,
    "INSTABILITY_WEIGHT": 0.4,
    "DISTURBANCE_WEIGHT": 0.2,
    "RECOVERY_BOOST": 1.2,  # 20% extra during recovery
    
    # PREFALL reflex settings
    "PREFALL_MIN": 0.15,      # Low risk gain
    "PREFALL_RANGE": 0.45,    # prefall_gain = PREFALL_MIN + PREFALL_RANGE * risk
    "PREFALL_MAX": 0.65,     # Maximum clamp
    
    # SAFE override settings
    "SAFE_THRESHOLD": 0.75,   # Catastrophic risk threshold
    "SAFE_GAIN": 0.12,        # 12% blend when active
}
```

## Implementation

All three adaptive functions use this configuration:
- `apply_edon_gain()` - Reads from `EDON_V31_HS_V4_CFG`
- `apply_prefall_reflex()` - Reads from `EDON_V31_HS_V4_CFG`
- `apply_safe_override()` - Reads from `EDON_V31_HS_V4_CFG`

## Usage

```bash
# Run validation test (100 episodes)
python run_eval.py --mode baseline --episodes 100 --profile high_stress \
    --output results/baseline_high_stress_v31_v4_100ep.json

python run_eval.py --mode edon --episodes 100 --profile high_stress \
    --edon-gain 0.75 --edon-controller-version v3 \
    --output results/edon_high_stress_v31_v4_100ep.json

# Validate results
python validate_v4_100ep.py
```

## Comparison Guidelines

- ✅ **DO**: Create new versions (V5, V6, etc.) and compare against V4
- ❌ **DON'T**: Modify `EDON_V31_HS_V4_CFG` parameters
- ✅ **DO**: Document any changes in new version files
- ❌ **DON'T**: Remove or rename V4 configuration

## Files

- **Config**: `evaluation/edon_controller_v3.py` (lines ~47-70)
- **Documentation**: `evaluation/EDON_V31_HS_V4_CONFIG.md`
- **Reference**: `EDON_V31_HS_V4_REFERENCE.md`
- **Validator**: `validate_v4_100ep.py`

