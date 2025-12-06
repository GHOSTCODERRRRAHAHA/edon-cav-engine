# What Training the v8 Strategy Policy Will Do

## Current State (3 Episodes = Near-Random)

**Right now, the policy:**
- Randomly selects strategies (NORMAL, HIGH_DAMPING, RECOVERY_BALANCE, COMPLIANT_TERRAIN)
- Outputs random modulations (gain_scale, lateral_compliance, step_height_bias)
- Doesn't understand when to use which strategy
- Doesn't learn from experience
- **Result:** Performs worse than baseline (-8.4%)

---

## After 300+ Episodes of Training

### 1. **Learns Strategy Selection**

The policy will learn **WHEN** to use each strategy:

- **NORMAL**: Use when stable (low tilt, low fail_risk)
- **HIGH_DAMPING**: Use when fail_risk is high or tilt is increasing
- **RECOVERY_BALANCE**: Use when recovering from high tilt
- **COMPLIANT_TERRAIN**: Use when on uneven terrain

**Example learned behavior:**
```
If fail_risk > 0.6:
    → Select HIGH_DAMPING strategy
    → Increase damping to prevent failure
```

### 2. **Learns Modulation Signals**

The policy will learn **HOW** to modulate actions:

- **gain_scale** (0.5 to 1.5): 
  - Low (0.5-0.8): When fail_risk is high → reduce action magnitude
  - High (1.2-1.5): When stable → allow more aggressive control

- **lateral_compliance** (0 to 1):
  - Low (0-0.3): When lateral tilt is high → reduce lateral movement
  - High (0.7-1.0): When stable → allow normal lateral movement

- **step_height_bias** (-1 to 1):
  - Negative: Lower step height when unstable
  - Positive: Normal step height when stable

**Example learned behavior:**
```
If tilt_mag > 0.2 and fail_risk > 0.5:
    → gain_scale = 0.6 (reduce actions by 40%)
    → lateral_compliance = 0.2 (reduce lateral movement by 80%)
    → step_height_bias = -0.5 (lower steps)
```

### 3. **How It Affects Control**

The flow is:
1. **Policy outputs** → strategy_id + modulations
2. **Reflex controller receives** → modulations + fail_risk + current state
3. **Reflex controller adjusts** → baseline action using:
   - Damping (from tilt, velocity, fail_risk)
   - Strategy modulations (gain_scale, lateral_compliance, step_height_bias)
4. **Final action** → applied to environment

**What the policy learns:**
- When to reduce action magnitude (via gain_scale)
- When to reduce lateral movement (via lateral_compliance)
- When to adjust step height (via step_height_bias)
- Which strategy to use for current situation

### 4. **Expected Improvements**

After 300+ episodes, the policy should learn to:

✅ **Reduce interventions by 5-15%**
   - Learns to detect high-risk situations early
   - Applies HIGH_DAMPING strategy before failure
   - Reduces action magnitude when fail_risk is high

✅ **Improve stability by 5-10%**
   - Learns to use appropriate damping
   - Adjusts gain_scale based on tilt
   - Uses lateral_compliance to prevent lateral falls

✅ **Increase episode length**
   - Prevents early failures
   - Maintains stability longer
   - Recovers from high-tilt situations

✅ **Better fail-risk utilization**
   - Uses fail_risk signal to trigger preventive actions
   - Learns to trust fail_risk predictions
   - Applies modulations before failure occurs

### 5. **Learning Process (PPO)**

During training, the policy learns through:

1. **Exploration**: Tries different strategies and modulations
2. **Reward signal**: Gets positive reward for:
   - Fewer interventions
   - Better stability
   - Longer episodes
   - Lower fail_risk
3. **Policy updates**: Adjusts network weights to maximize reward
4. **Convergence**: After 300+ episodes, policy converges to good behavior

### 6. **Concrete Example**

**Before training (random):**
```
State: tilt=0.25, fail_risk=0.7
Policy: Randomly selects NORMAL, gain_scale=1.2
Result: Action too aggressive → intervention occurs
```

**After training (learned):**
```
State: tilt=0.25, fail_risk=0.7
Policy: Selects HIGH_DAMPING, gain_scale=0.6, lateral_compliance=0.3
Result: Action reduced → intervention prevented
```

---

## What You'll See During Training

**Early episodes (1-50):**
- High policy loss (learning)
- Random strategies
- High entropy (exploring)
- Poor performance

**Mid episodes (50-150):**
- Policy loss decreasing
- Strategies becoming more consistent
- Entropy decreasing (exploiting)
- Performance improving

**Late episodes (150-300):**
- Policy loss stabilizing
- Strategies well-learned
- Low entropy (exploiting learned behavior)
- Good performance (5-15% better than baseline)

---

## Bottom Line

**Training the policy for 300+ episodes will:**
1. Teach it **WHEN** to use each strategy
2. Teach it **HOW** to modulate actions
3. Make it **SMARTER** about preventing failures
4. Result in **5-15% improvement** over baseline

**Without training:** Policy is random → worse than baseline
**With training:** Policy is learned → better than baseline

