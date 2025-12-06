# EDON Robot Stability API (v8 Integration)

**Version**: v1.0  
**Last Updated**: 2025-01-XX  
**Contact:** Charlie Biggins - charlie@edoncore.com

Complete guide for using EDON's robot stability control API.

---

## Overview

The Robot Stability API provides **real-time robot stability control** to prevent interventions and maintain balance.

**Key Features:**
- ✅ 97% intervention reduction (validated)
- ✅ Real-time control (<25ms latency)
- ✅ Historical context support (optional history parameter)
- ✅ Intervention risk prediction

---

## API Endpoint

**POST** `/oem/robot/stability`

**Content-Type:** `application/json`

---

## Request Format

```json
{
  "robot_state": {
    "roll": 0.05,
    "pitch": 0.02,
    "roll_velocity": 0.1,
    "pitch_velocity": 0.05,
    "com_x": 0.0,
    "com_y": 0.0
  },
  "history": [
    {
      "roll": 0.04,
      "pitch": 0.01,
      "roll_velocity": 0.08,
      "pitch_velocity": 0.04,
      "com_x": 0.0,
      "com_y": 0.0
    }
  ]
}
```

### Request Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `robot_state` | object | ✅ Yes | Current robot state |
| `robot_state.roll` | float | ✅ Yes | Roll angle (radians) |
| `robot_state.pitch` | float | ✅ Yes | Pitch angle (radians) |
| `robot_state.roll_velocity` | float | ✅ Yes | Roll angular velocity (rad/s) |
| `robot_state.pitch_velocity` | float | ✅ Yes | Pitch angular velocity (rad/s) |
| `robot_state.com_x` | float | No | Center of mass X position (default: 0.0) |
| `robot_state.com_y` | float | No | Center of mass Y position (default: 0.0) |
| `history` | array | No | Previous robot states (max 8, for temporal memory) |

---

## Response Format

```json
{
  "strategy_id": 1,
  "strategy_name": "HIGH_DAMPING",
  "modulations": {
    "gain_scale": 0.77,
    "compliance": 0.86,
    "bias": [-0.89, -0.89, -0.89, -0.89, -0.89, -0.89, -0.89, -0.89, -0.89, -0.89]
  },
  "intervention_risk": 0.01,
  "latency_ms": 24.12
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `strategy_id` | int | Strategy ID [0-3]: 0=NORMAL, 1=HIGH_DAMPING, 2=RECOVERY_BALANCE, 3=COMPLIANT_TERRAIN |
| `strategy_name` | string | Strategy name |
| `modulations.gain_scale` | float | Gain scale multiplier [0.5-2.0] |
| `modulations.compliance` | float | Lateral compliance [0.0-1.0] |
| `modulations.bias` | array | Step height bias vector (action space size) |
| `intervention_risk` | float | Predicted intervention risk [0.0-1.0] |
| `latency_ms` | float | Processing latency in milliseconds |

---

## Python SDK Usage

### Basic Usage

```python
from edon import EdonClient

# Initialize client
client = EdonClient(base_url="http://localhost:8002")

# Get robot stability control
robot_state = {
    "roll": 0.05,
    "pitch": 0.02,
    "roll_velocity": 0.1,
    "pitch_velocity": 0.05,
    "com_x": 0.0,
    "com_y": 0.0
}

result = client.robot_stability(robot_state)

print(f"Strategy: {result['strategy_name']}")
print(f"Gain Scale: {result['modulations']['gain_scale']:.2f}")
print(f"Intervention Risk: {result['intervention_risk']:.3f}")
```

### With Temporal Memory

```python
# Maintain history for temporal memory
history = []

while True:
    # Get current robot state
    current_state = get_robot_state()
    
    # Get stability control (with history)
    result = client.robot_stability(current_state, history=history[-8:])
    
    # Apply modulations to robot controller
    apply_modulations(
        gain_scale=result['modulations']['gain_scale'],
        compliance=result['modulations']['compliance'],
        bias=result['modulations']['bias']
    )
    
    # Update history (keep last 8 frames)
    history.append(current_state)
    if len(history) > 8:
        history.pop(0)
```

---

## Integration Example

### Complete Control Loop

```python
from edon import EdonClient
import time

# Initialize clients
edon_client = EdonClient(base_url="http://localhost:8002")

# Control loop
history = []
while True:
    # 1. Get robot state
    robot_state = {
        "roll": get_roll(),
        "pitch": get_pitch(),
        "roll_velocity": get_roll_velocity(),
        "pitch_velocity": get_pitch_velocity(),
        "com_x": get_com_x(),
        "com_y": get_com_y()
    }
    
    # 2. Get stability control
    stability = edon_client.robot_stability(robot_state, history=history[-8:])
    
    # 3. Get baseline action
    baseline_action = baseline_controller(robot_state)
    
    # 4. Apply modulations
    final_action = baseline_action * stability['modulations']['gain_scale']
    final_action += stability['modulations']['bias']
    
    # 5. Apply to robot
    robot.set_action(final_action)
    
    # 6. Update history
    history.append(robot_state)
    if len(history) > 8:
        history.pop(0)
    
    # 7. Control frequency (adjust based on your system)
    time.sleep(0.01)  # 100Hz
```

---

## Strategies

| ID | Name | Description | Use Case |
|----|------|-------------|----------|
| 0 | NORMAL | Standard control | Normal operation |
| 1 | HIGH_DAMPING | Increased damping | Oscillations detected |
| 2 | RECOVERY_BALANCE | Recovery mode | High instability |
| 3 | COMPLIANT_TERRAIN | Compliant terrain | Uneven surfaces |

---

## Error Handling

### 503 Service Unavailable

**Cause:** v8 models not loaded

**Solution:** Ensure models are available:
- `models/edon_v8_strategy_memory_features.pt`
- `models/edon_fail_risk_v1_fixed_v2.pt`

### 500 Internal Server Error

**Cause:** Invalid request or processing error

**Solution:** Check request format and server logs

---

## Performance

- **Latency:** <25ms typical
- **Throughput:** 40+ requests/second
- **Memory:** ~50MB per session (temporal buffers)

---

## Health Check

Check if robot stability is available:

```python
health = client.health()
if health.get("v8_robot_stability", {}).get("available"):
    print("Robot stability API is ready")
else:
    print("Robot stability API is not available")
```

---

## Best Practices

1. **Maintain History:** Use temporal memory (8 frames) for better predictions
2. **Error Handling:** Always check `intervention_risk` and handle high values
3. **Latency Monitoring:** Monitor `latency_ms` to ensure real-time performance
4. **Strategy Selection:** Use `strategy_id` to adjust control behavior
5. **Fail-Safe:** Have a fallback controller if API is unavailable

---

## See Also

- [OEM Integration Guide](OEM_INTEGRATION.md) - Complete integration guide
- [OEM Onboarding](OEM_ONBOARDING.md) - Getting started
- [API Contract](OEM_API_CONTRACT.md) - Full API specification

---

*Last Updated: After adding v8 robot stability to EDON Core*

