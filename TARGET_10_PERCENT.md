# Targeting 10%+ Improvements - Aggressive Fixes Applied

## Current Status
- **Stage**: Stage 0 (1-5% improvements)
- **Best intervention**: +1.07% (gain 0.90)
- **Best stability**: +6.25% (gain 0.60) - but interventions worse (tradeoff)
- **Overall**: ~1% improvement
- **Target**: 10%+ consistent improvements (Stage 1+)

## Aggressive Fixes Applied (V2)

### 1. PD Gains - 80% Increase
- `KP_ROLL: 0.10 → 0.18` (80% increase)
- `KP_PITCH: 0.10 → 0.18` (80% increase)
- `KD_ROLL: 0.03 → 0.05` (67% increase)
- `KD_PITCH: 0.03 → 0.05` (67% increase)

### 2. PREFALL Corrections - 25% Stronger
- `PREFALL_BASE: 0.28 → 0.35` (35% base correction)
- `PREFALL_MIN: 0.15 → 0.20` (20% minimum)
- `PREFALL_MAX: 0.40 → 0.50` (50% maximum)

### 3. SAFE Corrections - 2.4x Stronger
- `SAFE_GAIN: 0.015 → 0.025` (2.5% base)
- `safe_correction roll/pitch: -0.05 → -0.12` (2.4x multiplier)

### 4. Mode Multipliers - Increased
- **BRACE**: `prefall_mult: 1.25 → 1.40`, `safe_mult: 1.30 → 1.50`
- **ESCALATE**: `prefall_mult: 2.4 → 2.8`, `safe_mult: 2.7 → 3.0`
- **RECOVERY**: `prefall_mult: 2.0 → 2.5`, `safe_mult: 2.2 → 2.8`
- **All modes**: Less baseline damping (base_scale increased)

### 5. Target Ratios - Doubled/Tripled
- **SAFE**: `0.01 → 0.02` (2% base, was 1%)
- **PREFALL Low**: `0.15 → 0.25` (25%, was 15%)
- **PREFALL Medium**: `0.20 → 0.30` (30%, was 20%)
- **PREFALL High**: `0.25 → 0.35` (35%, was 25%)
- **FAIL**: `0.04 → 0.08` (8%, was 4%)
- **Caps**: SAFE max 4% (was 2%), PREFALL max 50% (was 27%), FAIL max 12% (was 5%)

## Expected Impact

These changes represent a **2-3x increase** in correction strength:
- **2.4x stronger SAFE corrections**
- **1.8x stronger PD gains**
- **1.25x stronger PREFALL base**
- **1.4-1.5x stronger mode multipliers**
- **2x higher target ratios**

**Combined effect**: Should push improvements from ~1% to **10%+**

## Test Results

Initial 15-episode test:
- Baseline: [checking...]
- EDON: [checking...]
- Improvements: [TBD]

## Next Steps

1. ✅ All aggressive fixes applied
2. ⏳ Running 30-episode validation test
3. ⏳ If <10%, consider:
   - Further increase PREFALL_BASE to 0.40-0.45
   - Increase PD gains to 0.20-0.25
   - Add correction damping if oscillations occur
4. ⏳ Run full comprehensive test suite once validated

## Files Modified
- `evaluation/edon_controller_v3.py` - All aggressive fixes applied

