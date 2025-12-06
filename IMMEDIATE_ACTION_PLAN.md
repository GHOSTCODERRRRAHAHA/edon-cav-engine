# Immediate Action Plan - Get to Stage 0 (+1% to +3%)

## Current Status

- **V5.1** (LPF only): +0.2% (best so far)
- **V5.2** (LPF + predicted boost): -1.4% (hurting)
- **Target**: +1% to +3%

## Immediate Actions (This Week)

### 1. Remove Predicted Boost (Go Back to V5.1)
**Why**: Predicted boost is making things worse (-1.4% vs +0.2%)
**Action**: 
- Remove predicted boost code
- Keep only LPF (V5.1 configuration)
- This gives us +0.2% baseline

### 2. Test LPF on Medium Stress
**Why**: V5.1 was tested on high_stress, need to verify on medium
**Action**:
- Run V5.1 (LPF only) on medium_stress
- 5 seeds × 30 episodes
- See if we get +0.2% to +0.5%

### 3. Optimize PREFALL Range Incrementally
**Why**: PREFALL may be too aggressive or not aggressive enough
**Action**:
- Test different PREFALL ranges:
  - 0.10-0.50 (more conservative)
  - 0.12-0.55 (slightly more)
  - 0.18-0.65 (more aggressive)
- Find sweet spot for medium_stress

### 4. Fine-tune LPF Alpha
**Why**: Current alpha (0.75 - 0.15) gives +0.2%, may need adjustment
**Action**:
- Test slight variations:
  - 0.73 - 0.15 (slightly less smoothing)
  - 0.77 - 0.15 (slightly more smoothing)
  - 0.75 - 0.13 (less aggressive scaling)
  - 0.75 - 0.17 (more aggressive scaling)

### 5. Verify EDON State Mapping
**Why**: May be using wrong signals or mapping incorrectly
**Action**:
- Check what EDON actually outputs
- Verify state mapping is correct
- Ensure we're using right signals for corrections

## Expected Outcome

After these actions:
- **Baseline**: +0.2% (V5.1 LPF)
- **After PREFALL tuning**: +0.5% to +1.0%
- **After LPF fine-tuning**: +1.0% to +1.5%
- **After state mapping fix**: +1.5% to +3.0%

**Target**: +1% to +3% ✅

## Next Steps After Stage 0

Once we hit +1% to +3%:
1. Lock that configuration
2. Test across all profiles
3. Then work on Stage 1 improvements

## Key Principle

**Incremental, systematic improvements.**
- Don't try to jump to 20% directly
- Get to Stage 0 first (+1% to +3%)
- Then build from there

