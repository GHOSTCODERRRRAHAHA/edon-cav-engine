# EDON Core vs v8: The Gap for OEMs

## The Question

**"Do we not have a v8-like system already in EDON Core? Because OEMs will use that API."**

## The Answer: **NO - There's a Gap**

EDON Core does **NOT** have v8-like robot stability control capabilities. They serve different purposes:

---

## What OEMs Get from EDON Core

### Current EDON Core API (`/oem/cav/batch`)

**Input:**
- Physiological sensor windows (240 samples of EDA, BVP, TEMP, ACC)
- Environmental context (temp_c, humidity, aqi, local_hour)

**Output:**
```json
{
  "state": "balanced" | "focus" | "restorative" | "overload",
  "cav_raw": 0-10000,
  "cav_smooth": 0-10000,
  "parts": {
    "p_stress": 0.0-1.0,
    "bio": 0.0-1.0,
    "env": 0.0-1.0,
    "circadian": 0.0-1.0
  },
  "controls": {
    "speed": 0.0-1.0,    // Recommended speed scale
    "torque": 0.0-1.0,   // Recommended torque scale
    "safety": 0.0-1.0    // Recommended safety margin
  }
}
```

**What This Does:**
- Predicts human operator/occupant cognitive state
- Recommends control scales to adapt robot behavior to human state
- **Does NOT**: Output direct robot control actions or process robot stability

---

## What v8 Provides (Research Platform)

### v8 Strategy Policy

**Input:**
- Robot state (roll, pitch, roll_velocity, pitch_velocity, COM position)
- Temporal memory (8 frames of history)
- Early-warning features (rolling variance, oscillation energy)
- Fail-risk prediction

**Output:**
```python
{
  "strategy_id": 0-3,  # Strategy selection
  "modulations": {
    "gain_scale": 0.5-2.0,      # Direct control modulation
    "compliance": 0.0-1.0,      # Direct control modulation
    "bias": np.array([...])      # Direct control modulation
  }
}
```

**What This Does:**
- Prevents robot interventions (97% reduction: 40.30 → 1.00 interventions/episode)
- Maintains robot stability in real-time
- **Does NOT**: Process physiological sensors or predict human state

---

## The Gap

### What OEMs Need

OEMs need **robot stability control** to prevent interventions, but EDON Core only provides:

1. ✅ **Human state prediction** (from physiological sensors)
2. ✅ **Control scale recommendations** (speed, torque, safety)
3. ❌ **NOT robot stability control** (from robot state sensors)
4. ❌ **NOT direct control actions** (strategy + modulations)

### The Problem

**EDON Core cannot prevent robot interventions** because:
- It doesn't process robot state (roll, pitch, velocities)
- It doesn't output direct control modulations
- It's designed for human-robot interaction, not robot stability

**v8 can prevent interventions**, but:
- It's a research platform (not productized)
- It's not exposed via EDON Core API
- OEMs can't access it through the standard API

---

## Potential Solutions

### Option 1: Add v8 to EDON Core API

**Add new endpoint:** `POST /oem/robot/stability`

**Input:**
```json
{
  "robot_state": {
    "roll": 0.0,
    "pitch": 0.0,
    "roll_velocity": 0.0,
    "pitch_velocity": 0.0,
    "com_x": 0.0,
    "com_y": 0.0
  },
  "history": [...],  // 8 frames of history
  "fail_risk": 0.0
}
```

**Output:**
```json
{
  "strategy_id": 0-3,
  "modulations": {
    "gain_scale": 0.5-2.0,
    "compliance": 0.0-1.0,
    "bias": [0.0, 0.0, ...]
  },
  "intervention_risk": 0.0-1.0
}
```

**Pros:**
- OEMs get v8 capabilities through standard API
- Productized and versioned
- Can be deployed alongside human state prediction

**Cons:**
- Requires adding robot state processing to EDON Core
- Different domain (robot control vs. human state)
- May need separate model serving infrastructure

### Option 2: Keep v8 Separate, Provide Integration Guide

**Keep v8 as research platform**, but provide:
- Integration guide for OEMs to use v8 directly
- Docker image with v8 model
- Python SDK for v8

**Pros:**
- Keeps domains separate (human state vs. robot control)
- No changes to EDON Core API
- OEMs can choose to use v8 or not

**Cons:**
- OEMs need to integrate two systems
- Not unified API experience
- v8 remains research platform (not productized)

### Option 3: Hybrid Approach

**EDON Core provides both:**
1. Human state prediction (existing `/oem/cav/batch`)
2. Robot stability control (new `/oem/robot/stability`)

**Unified API:**
```python
# Human state
human_state = client.cav(physio_window)

# Robot stability
robot_control = client.robot_stability(robot_state)

# Combined usage
final_action = apply_edon_scales(
    robot_control["modulations"],
    human_state["controls"]
)
```

**Pros:**
- Single API for all EDON capabilities
- OEMs get both human adaptation and robot stability
- Unified versioning and deployment

**Cons:**
- More complex API surface
- Different domains in same service
- Requires careful architecture

---

## Recommendation

### For Immediate OEM Needs

**Option 2 (Keep Separate)** is best because:
1. **Different domains**: Human state vs. robot control are fundamentally different
2. **Different requirements**: Robot control needs real-time (<10ms), human state needs 60-second windows
3. **Different models**: v8 uses learned policy, EDON Core uses LightGBM classifier
4. **Separation of concerns**: Keeps research platform separate from product

### For Future Productization

**Option 3 (Hybrid)** could work if:
1. v8 is fully productized (not just research)
2. Unified deployment makes sense for OEMs
3. API can handle both domains cleanly

---

## Current State Summary

| Capability | EDON Core (OEM API) | v8 (Research) |
|------------|---------------------|---------------|
| **Human state prediction** | ✅ Yes | ❌ No |
| **Control scale recommendations** | ✅ Yes | ❌ No |
| **Robot stability control** | ❌ No | ✅ Yes |
| **Direct control modulations** | ❌ No | ✅ Yes |
| **Intervention prevention** | ❌ No | ✅ Yes (97% reduction) |
| **Productized** | ✅ Yes | ❌ No (research) |
| **OEM API access** | ✅ Yes | ❌ No |

---

## Conclusion

**EDON Core does NOT have v8-like capabilities** for robot stability control. OEMs currently get:
- ✅ Human state prediction
- ✅ Control scale recommendations
- ❌ **NOT robot stability control**

**To provide v8-like capabilities to OEMs**, you need to either:
1. Add robot stability endpoint to EDON Core API
2. Productize v8 separately with its own API
3. Provide integration guide for OEMs to use v8 directly

**The gap exists because v8 and EDON Core solve different problems:**
- **v8**: Robot stability (prevents interventions)
- **EDON Core**: Human-robot interaction (adapts to human state)

**They're complementary, not replaceable.**

---

*Last Updated: After identifying the gap between EDON Core API and v8 capabilities*

