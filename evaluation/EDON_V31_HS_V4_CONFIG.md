# EDON v3.1 High-Stress V4 Configuration (FROZEN)

## Status: LOCKED - Reference Configuration

This configuration is **frozen** and serves as a reference point for comparing future changes.

## Performance (30 episodes, high_stress)
- **Interventions**: +4.4% improvement
- **Stability**: +8.4% improvement
- **Average**: **+6.4%** improvement
- **EDON Gain**: 0.75 (optimal)

## Configuration Parameters

### Adaptive EDON Gain
- **Base gain**: 0.4
- **Instability weight**: 0.4
- **Disturbance weight**: 0.2
- **Recovery boost**: 1.2 (20% extra during recovery)
- **Gain range**: 0.3 to 1.1

### Dynamic PREFALL Reflex
- **Minimum gain** (low risk): 0.15
- **Range**: 0.45 (added to minimum based on risk)
- **Maximum gain** (high risk): 0.60 (clamped at 0.65)
- **Formula**: `prefall_gain = 0.15 + 0.45 * fall_risk`

### SAFE Override
- **Activation threshold**: 0.75 (catastrophic risk)
- **Gain when active**: 0.12 (12% blend)
- **Behavior**: Only activates when catastrophic_risk > 0.75

## Implementation

All parameters are defined in `EDON_V31_HS_V4_CFG` dictionary in `evaluation/edon_controller_v3.py`.

Functions use this configuration:
- `apply_edon_gain()` - State-aware adaptive gain
- `apply_prefall_reflex()` - Dynamic risk-based PREFALL
- `apply_safe_override()` - Last-resort SAFE activation

## Usage

This configuration is automatically used when `edon_controller_version="v3"` is specified in `run_eval.py`.

## Validation

**Status**: Pending 100-episode validation test to confirm stability.

## Notes

- This configuration achieved the best results in incremental testing
- All parameters are locked - do not modify without creating a new version
- For future improvements, create V5, V6, etc. and compare against this baseline

