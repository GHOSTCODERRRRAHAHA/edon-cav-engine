# Testing: Can EDON Core Do What v8 Did?

## Important Clarification

**EDON Core and v8 are different systems:**

- **v8**: Robot state → Control modulations (prevents robot interventions)
- **EDON Core**: Physiological sensors → Control scales (adapts to human state)

**They can't directly replace each other**, but we can test if EDON Core helps when used **WITH** v8.

---

## Test Plan

### Test 1: v8 Only (Baseline)
- Run robot episodes with v8 strategy policy
- Measure: interventions, stability, episode length
- **Expected**: 1.00 interventions/episode (from previous results)

### Test 2: v8 + EDON Core Control Scales
- Run robot episodes with v8 strategy policy
- **Also** use EDON Core control scales to modulate actions
- Measure: interventions, stability, episode length
- **Question**: Does EDON Core help or hurt?

### Test 3: EDON Core Only (If Possible)
- Try to use EDON Core control scales without v8
- **Challenge**: EDON Core needs physiological sensors, not robot state
- **Workaround**: Map robot state to synthetic physiological signals (for testing)

---

## How to Run the Test

### Prerequisites

1. **Start EDON Core Server**:
```bash
docker run -d --name edon-server -p 8002:8000 -p 50052:50051 edon-server:v1.0.1
```

2. **Install EDON Core SDK**:
```bash
pip install sdk/python/edon-*.whl
```

3. **Verify v8 Model Exists**:
```bash
ls models/edon_v8_strategy_memory_features.pt
```

### Run Test

```bash
python test_edon_core_with_robot_control.py
```

---

## Expected Results

### Scenario 1: EDON Core Helps
- **v8 Only**: 1.00 interventions/episode
- **v8 + EDON Core**: 0.5-0.8 interventions/episode
- **Conclusion**: EDON Core's control scales provide additional safety

### Scenario 2: EDON Core Has No Effect
- **v8 Only**: 1.00 interventions/episode
- **v8 + EDON Core**: 1.00 interventions/episode
- **Conclusion**: EDON Core doesn't help (v8 already optimal)

### Scenario 3: EDON Core Hurts
- **v8 Only**: 1.00 interventions/episode
- **v8 + EDON Core**: 1.5-2.0 interventions/episode
- **Conclusion**: EDON Core's control scales conflict with v8's modulations

---

## What This Test Proves

### If EDON Core Helps:
- **Conclusion**: EDON Core's control scales provide additional safety layer
- **Use Case**: Use both v8 (robot stability) + EDON Core (human-robot interaction)

### If EDON Core Has No Effect:
- **Conclusion**: v8 is already optimal for robot stability
- **Use Case**: Use v8 for robot control, EDON Core for human state prediction (separate)

### If EDON Core Hurts:
- **Conclusion**: EDON Core's control scales conflict with v8's modulations
- **Use Case**: Don't combine them, use separately

---

## Key Insight

**v8 and EDON Core solve different problems:**

- **v8**: "How is the robot moving?" → Prevents robot from falling
- **EDON Core**: "How is the human feeling?" → Adapts robot to human state

**They're complementary, not replaceable.**

This test will show if they can work together, not if EDON Core can replace v8.

---

*Test script: `test_edon_core_with_robot_control.py`*

