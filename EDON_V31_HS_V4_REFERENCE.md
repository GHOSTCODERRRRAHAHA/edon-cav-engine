# EDON v3.1 High-Stress V4 - Reference Configuration

## Status: FROZEN - Do Not Modify

This configuration is **locked** and serves as the reference baseline for all future improvements.

## Configuration Name
**EDON_v3.1_high_stress_V4**

## Performance Claims (30 episodes, high_stress)
- **Interventions**: +4.4% improvement
- **Stability**: +8.4% improvement
- **Average**: **+6.4%** improvement
- **EDON Gain**: 0.75 (optimal)

## Validation Status
- ⏳ **100-episode validation**: [Running...]
- **Purpose**: Confirm improvements are stable (4-9% band) and not variance

## Configuration Parameters (FROZEN)

### Adaptive EDON Gain (`apply_edon_gain`)
```python
BASE_GAIN = 0.4
INSTABILITY_WEIGHT = 0.4
DISTURBANCE_WEIGHT = 0.2
RECOVERY_BOOST = 1.2  # 20% extra during recovery
Gain range: 0.3 to 1.1
```

### Dynamic PREFALL Reflex (`apply_prefall_reflex`)
```python
PREFALL_MIN = 0.15      # Low risk gain
PREFALL_RANGE = 0.45    # Added based on risk
PREFALL_MAX = 0.65      # Maximum clamp
Formula: prefall_gain = 0.15 + 0.45 * fall_risk
```

### SAFE Override (`apply_safe_override`)
```python
SAFE_THRESHOLD = 0.75   # Catastrophic risk threshold
SAFE_GAIN = 0.12        # 12% blend when active
```

## Implementation Location
- **File**: `evaluation/edon_controller_v3.py`
- **Config dict**: `EDON_V31_HS_V4_CFG`
- **Functions**: `apply_edon_gain()`, `apply_prefall_reflex()`, `apply_safe_override()`

## Usage
```bash
python run_eval.py --mode edon --episodes 100 --profile high_stress \
    --edon-gain 0.75 --edon-controller-version v3 \
    --output results/edon_high_stress_v31_v4_100ep.json
```

## Comparison Guidelines
- **DO**: Create new versions (V5, V6, etc.) and compare against V4
- **DON'T**: Modify V4 parameters directly
- **DO**: Document any changes in new version files
- **DON'T**: Remove or rename V4 configuration

## Validation Commands
```bash
# Run 100-episode baseline
python run_eval.py --mode baseline --episodes 100 --profile high_stress \
    --output results/baseline_high_stress_v31_v4_100ep.json

# Run 100-episode EDON V4
python run_eval.py --mode edon --episodes 100 --profile high_stress \
    --edon-gain 0.75 --edon-controller-version v3 \
    --output results/edon_high_stress_v31_v4_100ep.json

# Validate results
python validate_v4_100ep.py
```

## Expected Validation Results
If improvements are stable:
- **Interventions**: 4-9% improvement
- **Stability**: 4-9% improvement
- **Average**: 4-9% improvement (Stage 1 range)

This confirms the configuration is not just a lucky roll but a stable improvement.

