# Adaptive Gains V6 - Final Push for 10%+

## Changes from V4 (Best: +6.4%)

### 1. Increased PREFALL Range
- **V4**: `prefall_gain = 0.15 + 0.45 * risk` (range: 0.15-0.60)
- **V6**: `prefall_gain = 0.20 + 0.50 * risk` (range: 0.20-0.70)
- Minimum: 0.15 → 0.20 (33% increase)
- Maximum: 0.60 → 0.70 (17% increase)
- Clamp: 0.65 → 0.75 (15% increase)

### 2. Increased Recovery Boost
- **V4**: `gain *= 1.2` (20% boost)
- **V6**: `gain *= 1.3` (30% boost)

## Expected Impact
- **Stronger PREFALL**: 0.20-0.70 range (was 0.15-0.60)
- **More recovery help**: 30% boost (was 20%)
- **Should push**: +6.4% → 10%+

## Test Results
- [Testing...]

## Configuration (V6)
- Base gain: 0.4
- Instability weight: 0.4
- Disturbance weight: 0.2
- PREFALL range: 0.20-0.70
- Recovery boost: 30%
- SAFE threshold: 0.75
- SAFE gain: 0.12

## Target
**10%+ average improvement** in both interventions AND stability

