# Adaptive Baselines Added - Continuously Moving & Adapting

## What You Asked About

You asked about **"adaptive baseline or whatever its called where its constantly moving and adapting"** - that's **EWMA (Exponential Weighted Moving Average)** baselines that continuously adapt over time, just like CAV adaptive memory does.

## What We Had Before

**Basic risk baseline:**
- Simple EWMA for intervention risk
- Updated every 20 records
- Not as sophisticated as CAV's adaptive baselines

## What We Added Now

### 1. **Strategy Success Rate Baselines (EWMA per strategy)**
- **Continuously adapting** baseline for each strategy's success rate
- Uses EWMA (α=0.3) - same as CAV adaptive memory
- Tracks: `success_rate_mu` (mean), `success_rate_var` (variance)
- **Adapts automatically** as strategies perform better/worse over time

### 2. **Overall Success Rate Baseline (EWMA)**
- **Continuously adapting** baseline for overall success rate
- Tracks: `success_rate_mu` (mean), `success_rate_std` (std)
- **Adapts automatically** as overall performance changes

### 3. **Enhanced Risk Baseline (EWMA)**
- Already had this, but now uses it more effectively
- **Continuously adapting** baseline for intervention risk
- Tracks: `risk_mu` (mean), `risk_std` (std)

## How It Works

### EWMA Formula (Same as CAV)
```
new_baseline = α * current_value + (1 - α) * previous_baseline
```

Where:
- **α = 0.3** (smoothing factor - same as CAV)
- **current_value** = recent mean/variance
- **previous_baseline** = previous EWMA baseline

### Continuous Adaptation

**Every 20 records:**
1. Compute recent statistics (last 50-100 records)
2. Update EWMA baselines using formula above
3. Baselines **continuously move** toward recent patterns
4. Older data has less influence (exponential decay)

### Example

**Strategy 0 (NORMAL) baseline:**
- Day 1: `success_rate_mu = 0.5` (unknown)
- Day 2: After 50 records, `success_rate_mu = 0.65` (learning it works well)
- Day 3: After 100 records, `success_rate_mu = 0.70` (refined)
- Day 4: After 200 records, `success_rate_mu = 0.68` (stabilized)

**The baseline is constantly moving and adapting!**

## How It's Used

### Strategy Adjustments
Instead of using raw success rate, we now use:
- **Adaptive baseline** (EWMA mean) for each strategy
- **Z-score** = (current_success_rate - baseline) / baseline_std
- Adjust modulations based on deviation from **moving baseline**

### Risk Adjustments
- **Adaptive baseline** (EWMA mean) for intervention risk
- **Z-score** = (current_risk - baseline) / baseline_std
- Adjust modulations based on deviation from **moving baseline**

## Benefits

### 1. **Continuously Adapting**
- Baselines move as environment changes
- Adapts to MuJoCo's specific patterns
- Not static - always learning

### 2. **Personalized**
- Each strategy has its own adaptive baseline
- Learns what's "normal" for each strategy
- Adapts to robot's specific behavior

### 3. **Robust to Noise**
- EWMA smooths out noise
- Recent data has more weight
- Older data gradually fades

### 4. **Same as CAV**
- Uses same EWMA approach as CAV adaptive memory
- Consistent architecture
- Proven to work

## Summary

✅ **Strategy baselines** - EWMA per strategy (continuously adapting)  
✅ **Success baseline** - EWMA overall success (continuously adapting)  
✅ **Risk baseline** - EWMA intervention risk (continuously adapting)  
✅ **Z-score adjustments** - Based on deviation from moving baselines  
✅ **Same as CAV** - Uses same EWMA approach (α=0.3)  

**Result:** Baselines are now **constantly moving and adapting**, just like CAV adaptive memory! The system learns what's "normal" for each strategy and adapts as patterns change.

