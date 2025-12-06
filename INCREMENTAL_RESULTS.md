# Incremental Testing Results - Finding Sweet Spot

## Baseline
- Interventions: 40.7/ep
- Stability: 0.0209

## Test Results

### Test 1: Moderate PD Only (KP=0.14, KD=0.04)
- Interventions: 40.3/ep (+1.1%)
- Stability: 0.0208 (+0.3%)
- **Average: +0.7%** ❌ (Need 10%+)

### Test 2: Moderate PD + PREFALL (KP=0.14, PREFALL=0.30)
- [Running...]

### Test 3: Stronger PD + PREFALL (KP=0.16, PREFALL=0.32)
- [Running...]

## Strategy
1. ✅ Test 1: +0.7% - Need more
2. ⏳ Test 2: Adding PREFALL boost
3. ⏳ Test 3: Stronger PD + PREFALL
4. ⏳ If still <10%, increase SAFE corrections
5. ⏳ Continue incrementing until 10%+ achieved

## Current Settings
- KP_ROLL/PITCH: 0.16 (60% above baseline)
- KD_ROLL/PITCH: 0.045 (50% above baseline)
- PREFALL_BASE: 0.32 (32% base correction)
- SAFE_GAIN: 0.020 (2%)
- SAFE_CORR: -0.08 (60% increase)

## Next Steps
- Continue testing incrementally
- If oscillations occur, dial back
- Find sweet spot where 10%+ without destabilizing

