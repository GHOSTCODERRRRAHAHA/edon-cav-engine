# V5 Architectural Upgrades

## Changes from V4

### 1. Stability-Weighted Low-Pass Filter ⭐
**Location**: Before final torque output in `apply_edon_regulation_v3()`

```python
alpha = clamp(0.85 - 0.3 * instability_score, 0.4, 0.9)
torque_cmd = alpha * torque_cmd_prev + (1 - alpha) * torque_cmd
```

**Benefits**:
- Kills noise
- Kills over-corrections
- Makes corrections coherent
- When stable → more smoothing (prevents twitch)
- When unstable → less smoothing (still responsive)

**Expected**: 3-8% gains instantly

### 2. Predicted Instability Pre-Boost
**Location**: In `apply_edon_gain()`

```python
predicted_instability = max(0.0, delta_ema * 10.0)  # Proxy for future instability
if predicted_instability > 0.1:
    gain += 0.15 * predicted_instability  # Pre-boost up to 15%
```

**Benefits**:
- Pre-boosts EDON gain before fall begins
- Reacts to predicted instability, not just current

### 3. PREFALL Decay
**Location**: In `apply_prefall_reflex()`

```python
# Decay factor: 0.85 (15% decay per step, decays to ~50% in 4 steps)
if target_prefall_gain > prev_gain:
    prefall_gain = prev_gain + 0.5 * (target_prefall_gain - prev_gain)  # Ramp up
else:
    prefall_gain = prev_gain * 0.85 + target_prefall_gain * 0.15  # Decay
```

**Benefits**:
- Prevents PREFALL from spiking and holding
- Smooth decay over 4-6 timesteps
- More stable corrections

### 4. SAFE Pre-Trigger
**Location**: In `apply_safe_override()`

```python
# Pre-trigger: light blend when risk > 0.6 (not just 0.75)
if catastrophic_risk > 0.6 and catastrophic_risk <= 0.75:
    safe_gain = 0.04  # 4% blend for pre-trigger
    return (1.0 - safe_gain) * torque_cmd + safe_gain * safe_posture_torque
```

**Benefits**:
- Triggers earlier (0.6 vs 0.75)
- Light blend (4% vs 12%)
- Gives EDON chance to stabilize before disaster

## Goal

**Target**: +3% to +5% with consistent sign across 5 seeds

Once achieved, then chase +10%.

## Testing

Run multi-seed validation:
```bash
python validate_v4_multi_seed.py
```

Expected improvement over V4 (-1.7% ± 1.6%):
- Low-pass filter should reduce variance
- Pre-boost should improve interventions
- PREFALL decay should stabilize corrections
- SAFE pre-trigger should prevent disasters

