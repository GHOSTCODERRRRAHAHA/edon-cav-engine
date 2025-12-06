# Aggressive Fixes V2 - Targeting 10%+ Improvements

## Goal
Move from **Stage 0 (1-5%)** to **Stage 1 (5-15%)** or better, targeting **10%+ consistent improvements**.

## Current Status
- Best intervention: +1.07% (gain 0.90)
- Best stability: +6.25% (gain 0.60) - but interventions worse (tradeoff)
- Overall: ~1% improvement (Stage 0)

## Aggressive Fixes Applied

### 1. **Increased PD Gains (80% increase)**
**Location**: Lines 36-39
- `KP_ROLL: 0.10 → 0.18` (80% increase)
- `KP_PITCH: 0.10 → 0.18` (80% increase)
- `KD_ROLL: 0.03 → 0.05` (67% increase)
- `KD_PITCH: 0.03 → 0.05` (67% increase)

**Impact**: Much stronger tilt corrections

### 2. **Increased PREFALL Correction Strength**
**Location**: Lines 51-53
- `PREFALL_BASE: 0.28 → 0.35` (35% base, was 28%)
- `PREFALL_MIN: 0.15 → 0.20` (20% minimum, was 15%)
- `PREFALL_MAX: 0.40 → 0.50` (50% maximum, was 40%)

**Impact**: 25% stronger PREFALL corrections

### 3. **Increased SAFE Correction Strength**
**Location**: Line 59, Lines 745-746
- `SAFE_GAIN: 0.015 → 0.025` (2.5% base, was 1.5%)
- `safe_correction[0/1]: -0.05 → -0.12` (2.4x stronger roll/pitch correction)

**Impact**: 67% stronger SAFE corrections

### 4. **Increased Mode Multipliers**
**Location**: Lines 486-502
- **BRACE**: 
  - `base_scale: 0.95 → 0.98` (less damping)
  - `prefall_mult: 1.25 → 1.40` (40% boost, was 25%)
  - `safe_mult: 1.30 → 1.50` (50% boost, was 30%)
- **ESCALATE**:
  - `base_scale: 0.65 → 0.75` (less damping)
  - `prefall_mult: 2.4 → 2.8` (17% stronger)
  - `safe_mult: 2.7 → 3.0` (11% stronger)
- **RECOVERY**:
  - `base_scale: 0.75 → 0.80` (less damping)
  - `prefall_mult: 2.0 → 2.5` (25% stronger)
  - `safe_mult: 2.2 → 2.8` (27% stronger)

**Impact**: Stronger corrections in all modes, less baseline damping

### 5. **Increased Target Ratios**
**Location**: Lines 272-303
- **SAFE**: `0.01 → 0.02` (2% base, was 1%)
- **PREFALL Low Risk**: `0.15 → 0.25` (25%, was 15%)
- **PREFALL Medium Risk**: `0.20 → 0.30` (30%, was 20%)
- **PREFALL High Risk**: `0.25 → 0.35` (35%, was 25%)
- **FAIL**: `0.04 → 0.08` (8%, was 4%)
- **Caps**: SAFE max 4% (was 2%), PREFALL max 50% (was 27%), FAIL max 12% (was 5%)

**Impact**: Much higher correction ratios across all zones

## Expected Impact

These aggressive changes should:
1. **2-3x stronger corrections** in PREFALL zones
2. **2.4x stronger SAFE corrections** when risk is elevated
3. **80% stronger PD gains** for tilt correction
4. **Less baseline damping** to preserve base controller strength
5. **Higher correction ratios** across all zones

**Target**: 10%+ improvements in both interventions AND stability (no tradeoffs)

## Risk

- May cause oscillations if corrections are too strong
- May destabilize if direction checking isn't perfect
- Need to monitor for over-correction

## Next Steps

1. Test with aggressive fixes
2. If oscillations occur, add damping to corrections
3. If still <10%, further increase PREFALL_BASE to 0.40-0.45
4. Run full comprehensive test suite once validated

