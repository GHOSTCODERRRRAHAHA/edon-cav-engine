# Detailed Roadmap to 20% Improvement

## Current State

- **Best**: V5.1 (LPF only) = +0.2% (high_stress, 3 seeds)
- **Target**: 20%
- **Gap**: 19.8 percentage points

## What We Need to Do - Stage by Stage

### Stage 0: +1% to +3% (Current Focus)

#### 1. Fix Predicted Boost (V5.2)
**Problem**: Predicted boost is making things worse (-1.4%)
**Solutions**:
- Try different predicted instability source (not just delta_ema)
- Use EDON's actual prediction if available
- Or remove predicted boost, keep only LPF

#### 2. Optimize PREFALL Scaling
**Current**: PREFALL range 0.15-0.60 (V4 config)
**Needs**:
- Test different PREFALL ranges incrementally
- Find optimal min/max for medium_stress
- Ensure PREFALL corrections are in right direction

#### 3. Improve State Mapping
**Current**: Using p_stress, p_chaos, risk_ema
**Needs**:
- Verify EDON state mapping is correct
- Check if we're using the right EDON outputs
- Ensure state transitions are smooth

#### 4. Fine-tune LPF
**Current**: alpha = 0.75 - 0.15 * instability (+0.2%)
**Needs**:
- Test if slightly different alpha helps
- May need profile-specific alpha formulas

**Expected**: +1% to +3% after these fixes

---

### Stage 1: +5% to +10%

#### 1. Better Correction Algorithms
**Current**: Simple PD on tilt
**Needs**:
- Multi-layer corrections (tilt + velocity + acceleration)
- Joint-specific corrections (not just balance joints)
- Adaptive PD gains based on state

#### 2. Improved State Awareness
**Current**: Basic state mapping
**Needs**:
- More granular state detection
- Better transition handling
- State-specific correction strategies

#### 3. Better Integration
**Current**: Additive corrections
**Needs**:
- More sophisticated blending
- Context-aware corrections
- Better coordination with baseline

#### 4. Enhanced Prediction
**Current**: delta_ema proxy
**Needs**:
- Use EDON's actual prediction capabilities
- Multi-step ahead prediction
- Better risk forecasting

**Expected**: +5% to +10% after these improvements

---

### Stage 2: +10% to +15%

#### 1. Advanced Control Strategies
**Needs**:
- Model Predictive Control (MPC) integration
- Optimal control theory
- Trajectory optimization

#### 2. Multi-Layer Architecture
**Needs**:
- Hierarchical control (high-level + low-level)
- Multiple correction layers
- Coordinated multi-joint control

#### 3. Better Disturbance Handling
**Needs**:
- Disturbance estimation
- Disturbance rejection
- Adaptive disturbance compensation

#### 4. Predictive Control
**Needs**:
- Look-ahead prediction
- Pre-emptive corrections
- Anticipatory control

**Expected**: +10% to +15% after these changes

---

### Stage 3: +15% to +20%

#### 1. Fundamental Architectural Improvements
**Needs**:
- Redesign control architecture
- Better state estimation
- Advanced filtering

#### 2. Machine Learning Integration
**Needs**:
- Learned correction policies
- Adaptive parameter tuning
- Experience-based improvements

#### 3. Optimal Control Integration
**Needs**:
- LQR/LQG controllers
- Optimal trajectory tracking
- Constrained optimization

#### 4. Advanced State Estimation
**Needs**:
- Kalman filtering
- Sensor fusion
- Better uncertainty handling

**Expected**: +15% to +20% after these fundamental changes

---

## Immediate Next Steps (Stage 0)

### Priority 1: Get LPF Working Consistently
- ✅ LPF is working (+0.2%)
- ⏳ Need to get it to +1% to +3%
- **Action**: Test LPF on medium_stress, optimize alpha

### Priority 2: Fix or Remove Predicted Boost
- ❌ Predicted boost is hurting (-1.4%)
- **Options**:
  - Remove it (go back to V5.1)
  - Fix it (better prediction source)
  - Make it even more conservative

### Priority 3: Optimize PREFALL
- ⏳ Test different PREFALL ranges
- ⏳ Ensure corrections are correct
- ⏳ Test on different profiles

### Priority 4: Improve State Mapping
- ⏳ Verify EDON outputs are correct
- ⏳ Check state transitions
- ⏳ Ensure we're using right signals

## Realistic Timeline

- **Stage 0** (+1% to +3%): 1-2 weeks (current focus)
- **Stage 1** (+5% to +10%): 1-2 months
- **Stage 2** (+10% to +15%): 3-6 months
- **Stage 3** (+15% to +20%): 6-12 months

## Key Insight

**We need incremental progress, not a magic bullet.**

Each stage builds on the previous:
- Stage 0: Parameter tuning + basic improvements
- Stage 1: Algorithm improvements
- Stage 2: Architectural improvements
- Stage 3: Fundamental redesign

**Focus on Stage 0 first, then work up.**

