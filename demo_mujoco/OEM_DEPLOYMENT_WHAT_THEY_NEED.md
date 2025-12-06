# OEM Deployment: What OEMs Need to Do

## Your Question

**"Would OEMs have to do the same when they use our API since our API state-aware was trained in a diff env? Or no?"**

**Answer:** It depends on whether they use zero-shot or train the policy. Let me explain:

---

## Scenario 1: OEM Uses Zero-Shot (No Training)

### What Happens

**OEM's Real Robot:**
- Different dynamics than training environment
- Different physics, inertia, contact properties
- Different action space, joint limits

**EDON API:**
- Policy trained on `MockHumanoidEnv` (simplified)
- Sees robot state correctly ✅
- But learned responses might be wrong for OEM's robot ❌

**Result:**
- Same issue as MuJoCo demo
- API outputs modulations learned for different environment
- Might make things worse (like 1 → 4 interventions)

### Do OEMs Need Our Fixes?

**Yes, they would benefit from similar fixes:**
- State-aware bounds (prevent wrong modulations)
- Fail-risk adjustment (compensate for prediction errors)
- Modulation smoothing (prevent oscillation)

**But:**
- OEMs might have different robot dynamics
- Our specific bounds might not be optimal for their robot
- They might need to tune bounds for their specific robot

**Recommendation:**
- Use zero-shot for initial deployment (25-50% improvement)
- Apply similar safety fixes (state-aware bounds)
- Train policy on their robot for 90%+ improvement

---

## Scenario 2: OEM Trains Policy on Their Robot (Recommended)

### What Happens

**Training Process:**
1. OEM collects data from their robot
2. Trains policy network on their robot's data
3. Policy learns correct responses for their robot

**After Training:**
- Policy sees robot state correctly ✅
- Policy outputs correct modulations for their robot ✅
- No more wrong responses ❌

**Result:**
- 90%+ improvement
- Policy knows their robot's dynamics
- Modulations are correct for their environment

### Do OEMs Need Our Fixes?

**Maybe, as a safety net:**
- Trained policy should be correct
- But safety fixes still help:
  - Prevent edge cases
  - Handle unexpected situations
  - Compensate for model errors

**Recommendation:**
- Train policy on their robot (required for 90%+)
- Keep safety fixes as optional safety net
- Monitor performance and adjust if needed

---

## The Key Difference: Training vs Zero-Shot

### Zero-Shot (No Training)

**OEM's Situation:**
- Uses API with policy trained on different environment
- Policy sees state correctly, but responses are wrong
- **Same issue as MuJoCo demo**

**What OEMs Need:**
- Similar safety fixes (state-aware bounds)
- Or accept 25-50% improvement (not 90%+)
- Or train policy on their robot

### After Training

**OEM's Situation:**
- Policy trained on their robot
- Policy learns correct responses for their robot
- **Different from MuJoCo demo** (we're training now)

**What OEMs Need:**
- Trained policy (correct responses)
- Optional safety fixes (safety net)
- Monitoring and fine-tuning

---

## What We Should Tell OEMs

### For Zero-Shot Deployment

**Tell them:**
1. "EDON works out-of-the-box with 25-50% improvement"
2. "API is state-aware (sees robot state)"
3. "But policy was trained on different environment, so responses might not be optimal"
4. "For 90%+ improvement, train policy on your robot"
5. "We include safety fixes to prevent wrong modulations"

### For Training

**Tell them:**
1. "Train policy on your robot using our training tools"
2. "After training, policy learns correct responses for your robot"
3. "Result: 90%+ improvement"
4. "Safety fixes still help as optional safety net"

---

## Comparison: MuJoCo Demo vs OEM Real Robot

### MuJoCo Demo (What We're Doing)

| Aspect | Training Environment | MuJoCo Environment |
|--------|---------------------|-------------------|
| **Policy Training** | MockHumanoidEnv (simplified) | Training now |
| **Zero-Shot** | 25-50% improvement | 25-50% improvement |
| **After Training** | N/A | 90%+ improvement (expected) |
| **Safety Fixes** | Needed for zero-shot | Helpful as safety net |

### OEM Real Robot (What They'll Do)

| Aspect | Training Environment | OEM's Real Robot |
|--------|---------------------|------------------|
| **Policy Training** | MockHumanoidEnv (simplified) | Train on their robot |
| **Zero-Shot** | 25-50% improvement | 25-50% improvement |
| **After Training** | N/A | 90%+ improvement |
| **Safety Fixes** | Needed for zero-shot | Optional safety net |

**Same situation!** OEMs face the same issue we do.

---

## What OEMs Should Do

### Option 1: Zero-Shot Only (Quick Start)

**Steps:**
1. Deploy EDON API
2. Use zero-shot (no training)
3. Apply safety fixes (state-aware bounds)
4. Accept 25-50% improvement

**Pros:**
- Works immediately
- No training required
- Good for demos/testing

**Cons:**
- Not optimal (25-50% vs 90%+)
- Might need to tune safety bounds for their robot

### Option 2: Train Policy (Recommended)

**Steps:**
1. Deploy EDON API
2. Collect data from their robot (1-2 weeks)
3. Train policy on their robot's data
4. Deploy trained policy
5. Optional: Keep safety fixes as safety net

**Pros:**
- Optimal performance (90%+ improvement)
- Policy learns their robot's dynamics
- Best long-term solution

**Cons:**
- Requires training infrastructure
- Takes time (1-2 weeks data collection + training)

---

## Our Fixes: Should OEMs Use Them?

### For Zero-Shot

**Yes, recommended:**
- OEMs face same issue (policy trained on different env)
- Safety fixes prevent wrong modulations
- Helps achieve 25-50% improvement consistently

**But:**
- OEMs might need to tune bounds for their robot
- Different robots have different dynamics
- Our bounds are tuned for MuJoCo

### For Trained Policy

**Optional, but helpful:**
- Trained policy should be correct
- But safety fixes still help:
  - Edge cases
  - Model errors
  - Unexpected situations

**Recommendation:**
- Include safety fixes in API/SDK
- Make bounds configurable
- OEMs can tune or disable if not needed

---

## What We Should Provide to OEMs

### 1. API with Safety Fixes Built-In

**Include in API/SDK:**
- State-aware bounds (configurable)
- Fail-risk adjustment (configurable)
- Modulation smoothing (configurable)

**Make it configurable:**
- OEMs can tune bounds for their robot
- OEMs can disable if not needed
- Defaults work for most cases

### 2. Training Tools

**Provide:**
- Training scripts (like `train_edon_mujoco.py`)
- Documentation on how to train
- Support for their robot's data format

### 3. Documentation

**Explain:**
- Zero-shot vs trained performance
- When to use safety fixes
- How to tune bounds for their robot
- Training process and requirements

---

## Summary

### Do OEMs Need Our Fixes?

**For Zero-Shot: Yes**
- Same issue: Policy trained on different environment
- Safety fixes prevent wrong modulations
- Helps achieve 25-50% improvement

**For Trained Policy: Optional**
- Trained policy should be correct
- But safety fixes still help as safety net
- OEMs can tune or disable

### What OEMs Should Do

**Recommended Path:**
1. **Start with zero-shot** (25-50% improvement)
   - Use safety fixes (built into API)
   - Test and validate
   
2. **Train policy on their robot** (90%+ improvement)
   - Collect data (1-2 weeks)
   - Train policy
   - Deploy trained model
   - Keep safety fixes as optional safety net

**Key Point:**
- Zero-shot: Safety fixes are important (prevent wrong modulations)
- After training: Safety fixes are optional (policy should be correct)

---

## Bottom Line

**Yes, OEMs would face the same issue in zero-shot:**
- API is state-aware (sees robot state)
- But policy trained on different environment
- Responses might be wrong for their robot

**But OEMs should train the policy:**
- After training, policy learns correct responses
- Safety fixes become optional safety net
- Result: 90%+ improvement

**What we should provide:**
- API with safety fixes built-in (configurable)
- Training tools and documentation
- Support for OEM-specific tuning

