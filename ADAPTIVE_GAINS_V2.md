# Adaptive Gains V2 - Tuned for 10%+ Target

## Changes from V1

### 1. Increased Base Gain
- **V1**: `base_gain = 0.4`
- **V2**: `base_gain = 0.5` (25% increase)

### 2. Increased Adaptive Weights
- **V1**: `gain = base_gain + 0.4 * instability + 0.2 * disturbance`
- **V2**: `gain = base_gain + 0.5 * instability + 0.3 * disturbance`
- Instability weight: 0.4 → 0.5 (25% increase)
- Disturbance weight: 0.2 → 0.3 (50% increase)

### 3. Increased PREFALL Range
- **V1**: `prefall_gain = 0.10 + 0.35 * risk` (range: 0.10-0.45)
- **V2**: `prefall_gain = 0.15 + 0.35 * risk` (range: 0.15-0.50)
- Minimum: 0.10 → 0.15 (50% increase)
- Maximum: 0.5 → 0.55 (10% increase)

### 4. Earlier SAFE Activation
- **V1**: `catastrophic_risk <= 0.8` (activates at 80%)
- **V2**: `catastrophic_risk <= 0.75` (activates at 75%)
- **V1**: `safe_gain = 0.08`
- **V2**: `safe_gain = 0.12` (50% increase)

## V1 Results (30 episodes)
- Interventions: +4.8%
- Stability: +6.8%
- **Average: +5.8%**

## V2 Expected Impact
- Stronger base corrections (base_gain 0.4 → 0.5)
- More responsive to instability/disturbance
- Stronger PREFALL at all risk levels
- Earlier and stronger SAFE activation

## Target
**10%+ average improvement** in both interventions AND stability

