# EDON Core vs v8: Feature Comparison

## The Question

**Does EDON Core have the same features as v8?**

**Short Answer: NO** - They're fundamentally different systems solving different problems, but they share **conceptual similarities**.

---

## What Each System Does

### EDON Core Engine
**Purpose**: Physiological state prediction from wearable sensors  
**Domain**: Human cognitive states (wearable devices → CAV scores → cognitive states)  
**Input**: 6 physiological features from 60-second sensor windows (EDA, BVP, accelerometer)  
**Output**: CAV score, state (restorative/balanced/focus/overload), p_stress, p_chaos

### EDON v8 Strategy Policy
**Purpose**: Robot control intervention prevention  
**Domain**: Humanoid robot stability (robot observations → strategy selection → control modulations)  
**Input**: 248-dim stacked observations (8 frames × 31 features)  
**Output**: Strategy ID, modulations (gain_scale, compliance, bias)

---

## Feature-by-Feature Comparison

### 1. Temporal Memory

#### v8 (Robot Control)
- **8-frame stacking**: Concatenates last 8 observation frames (248 dims)
- **Direct observation history**: Stores raw robot state (roll, pitch, velocities, COM)
- **Inline computation**: Computed in environment wrapper
- **Purpose**: See trends in robot stability over time

#### EDON Core (Physiological State)
- **24-hour rolling context**: Maintains last 24 hours of CAV responses
- **Hourly EWMA statistics**: Mean, variance, state distributions per hour
- **SQLite persistence**: Stores in database (7-day retention)
- **Purpose**: Learn personalized baselines for cognitive state prediction

**Similarity**: ✅ Both use temporal memory  
**Difference**: ❌ Different time scales (8 frames vs 24 hours), different data (robot state vs CAV scores)

---

### 2. Early-Warning Features

#### v8 (Robot Control)
- **Rolling variance** (4 features):
  - Variance of roll, pitch, roll_velocity, pitch_velocity over last 20 steps
  - Detects increasing instability trends
- **Oscillation energy** (1 feature):
  - Mean of squared angular velocities
  - Detects wobbling/oscillatory instability
- **Near-fail density** (1 feature):
  - Fraction of last 20 steps with fail_risk > 0.6
  - Detects persistent high-risk periods

#### EDON Core (Physiological State)
- **Z-score anomaly detection**:
  - Computes z-score of current CAV vs hourly baseline
  - Detects deviations from normal patterns
- **State frequency distribution**:
  - Tracks state probabilities per hour
  - Detects unusual state patterns
- **Environmental adjustments**:
  - AQI-based sensitivity adjustments
  - Time-of-day contextual adjustments

**Similarity**: ✅ Both detect anomalies/patterns  
**Difference**: ❌ Different features (robot instability vs cognitive state deviations), different methods (variance/oscillation vs z-scores)

---

### 3. Risk Assessment

#### v8 (Robot Control)
- **Fail-risk prediction model**:
  - Standalone neural network (15 → 1)
  - Predicts robot failure 0.5-1.0s ahead
  - Input: roll, pitch, velocities, COM, fail_risk history
  - Output: fail_risk ∈ [0, 1]

#### EDON Core (Physiological State)
- **p_stress probability**:
  - Component of CAV computation
  - Predicts stress state from physiological signals
- **p_chaos probability**:
  - Derived from state detection
  - Indicates chaotic/unstable cognitive state
- **CAV score**:
  - Composite score indicating cognitive availability
  - Lower = more stressed/overloaded

**Similarity**: ✅ Both predict risk/stress  
**Difference**: ❌ Different risks (robot failure vs cognitive overload), different models (standalone NN vs CAV engine)

---

### 4. State Detection

#### v8 (Robot Control)
- **Phase detection**:
  - "stable", "warning", "recovery", "prefall", "fail"
  - Based on fail_risk and instability_score
  - Used for strategy selection

#### EDON Core (Physiological State)
- **State classification**:
  - "restorative", "balanced", "focus", "overload"
  - Based on LightGBM classifier (6 features → state)
  - Used for CAV computation

**Similarity**: ✅ Both classify states  
**Difference**: ❌ Different states (robot stability vs cognitive states), different methods (rule-based vs ML classifier)

---

### 5. Adaptive Control

#### v8 (Robot Control)
- **Strategy selection** (4 discrete options):
  - NORMAL_BALANCE, RECOVERY_BALANCE, AGGRESSIVE_BALANCE, EMERGENCY_STOP
- **Continuous modulations**:
  - gain_scale, compliance, bias
  - Modulates baseline controller

#### EDON Core (Physiological State)
- **Adaptive adjustments**:
  - Sensitivity multiplier (1.0-1.25x)
  - Environment weight adjustment (0.8-1.0x)
  - Based on z-scores and AQI patterns
- **Control scales** (for OEM integration):
  - Speed, torque, safety margins
  - Adaptive gain based on state

**Similarity**: ✅ Both adapt based on context  
**Difference**: ❌ Different adaptations (robot control vs cognitive state sensitivity), different outputs (control modulations vs CAV adjustments)

---

## Why They're Different

### Different Domains

| Aspect | EDON Core | v8 |
|--------|-----------|-----|
| **Input** | Wearable sensor data (EDA, BVP, accel) | Robot state (roll, pitch, velocities, COM) |
| **Output** | CAV score, cognitive state | Strategy ID, control modulations |
| **Time Scale** | 24-hour rolling context | 8-frame stacking (~1-2 seconds) |
| **Purpose** | Predict human cognitive state | Prevent robot interventions |
| **Domain** | Physiological monitoring | Robot control |

### Different Implementations

| Feature | EDON Core | v8 |
|---------|-----------|-----|
| **Temporal Memory** | 24-hour EWMA, SQLite | 8-frame deque, in-memory |
| **Early Warning** | Z-scores, state frequencies | Rolling variance, oscillation energy |
| **Risk Assessment** | p_stress, p_chaos, CAV | Fail-risk model (standalone NN) |
| **State Detection** | LightGBM classifier | Rule-based phase detection |
| **Adaptation** | Sensitivity, env weights | Strategy selection, modulations |

---

## Conceptual Similarities

Despite being different systems, they share **architectural concepts**:

1. **Temporal Memory**: Both maintain history to see patterns
2. **Early Warning**: Both detect problems before they become critical
3. **Risk Assessment**: Both predict undesirable states
4. **State Detection**: Both classify current state
5. **Adaptive Control**: Both adjust behavior based on context

**This is why v8 validates concepts for EDON Core** - they share the same architectural principles, even though the implementations are different.

---

## Why v8 Doesn't Use EDON Core

### Design Decision

**v8 is the research platform** that validates concepts **before** they're productized:

1. **Different Domains**: v8 needs robot-specific features (rolling variance, oscillation energy), not physiological features (EDA, BVP)
2. **Different Time Scales**: v8 needs millisecond-level control (8 frames), not hour-level baselines (24 hours)
3. **Experimental Control**: v8 needs full control to test new ideas inline
4. **Rapid Iteration**: Inline code allows fast experimentation

### What v8 Validates for EDON Core

v8 proves that **temporal memory + early-warning features + risk assessment** enable predictive control. These concepts are then productized in EDON Core for physiological state prediction.

---

## Summary

### Does EDON Core Have the Same Features as v8?

**NO** - They have **conceptually similar features** but **different implementations**:

| Feature Type | v8 | EDON Core | Same? |
|--------------|----|-----------|-------|
| **Temporal Memory** | 8-frame stacking | 24-hour rolling context | ❌ Different |
| **Early Warning** | Rolling variance, oscillation | Z-scores, state frequencies | ❌ Different |
| **Risk Assessment** | Fail-risk model | p_stress, p_chaos | ❌ Different |
| **State Detection** | Phase (stable/warning/fail) | State (restorative/balanced/focus/overload) | ❌ Different |
| **Adaptation** | Strategy + modulations | Sensitivity + env weights | ❌ Different |

### The Key Insight

**v8 and EDON Core share architectural principles** (temporal memory, early warning, risk assessment) but **solve different problems** (robot control vs physiological state prediction). This is why v8 validates concepts for EDON Core - they prove the architecture works, even though the implementations are domain-specific.

---

*Last Updated: After analyzing EDON Core vs v8 feature differences*

