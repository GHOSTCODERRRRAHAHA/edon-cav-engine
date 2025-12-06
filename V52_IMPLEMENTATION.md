# V5.2 Implementation Details

## Code Changes

### 1. Predicted Instability (in `apply_edon_regulation_v3()`)

**Before (V5.1)**:
```python
# No predicted instability
edon_state_dict = {
    "instability_score": instability_score,
    ...
}
```

**After (V5.2)**:
```python
# Super-conservative: delta_ema * 2.0, capped at 0.5
predicted_instability = max(0.0, min(delta_ema * 2.0, 0.5))

edon_state_dict = {
    "instability_score": instability_score,
    ...
    "predicted_instability": predicted_instability  # V5.2
}
```

### 2. Gain Modulation (in `apply_edon_gain()`)

**Before (V5.1)**:
```python
gain = base_gain + 0.4 * instability + 0.2 * disturbance
# No predicted boost
```

**After (V5.2)**:
```python
gain = base_gain + 0.4 * instability + 0.2 * disturbance
# Mild predicted boost: up to +10%
gain *= (1.0 + 0.2 * predicted_instability)  # 0-10% bump
```

### 3. LPF (Unchanged)

**V5.1 and V5.2**:
```python
alpha = clamp(0.75 - 0.15 * instability_score, 0.4, 0.9)
torque_cmd = alpha * torque_cmd_prev + (1.0 - alpha) * torque_cmd
```

## Parameters

- **Predicted instability scale**: 2.0 (was 10.0 in initial attempt)
- **Predicted instability cap**: 0.5 (was 1.0)
- **Boost multiplier**: 0.2 (gives max 10% boost when predicted=0.5)
- **LPF alpha**: 0.75 - 0.15 * instability (unchanged from V5.1)

## Expected Behavior

- **When stable** (delta_ema ≈ 0): predicted_instability ≈ 0, no boost
- **When risk increasing** (delta_ema > 0): predicted_instability increases, mild boost
- **Max boost**: 10% (when predicted_instability = 0.5, delta_ema = 0.25)

## Validation Criteria

**Good**:
- Mean: +1% to +3%
- Std: ±1-2%
- Most seeds positive

**Bad**:
- Wild mix of +4% and -5%
- High variance (±3-4%)
- Inconsistent signs

