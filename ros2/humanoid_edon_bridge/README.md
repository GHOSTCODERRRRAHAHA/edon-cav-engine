# EDON for Humanoids 🦾 — Emotional & Context Nervous System

EDON turns raw biosignals + environment into a high-level emotional/context state that humanoids can use to move, speak, look, and shape the environment in a more human-aligned way.

Instead of every robot team reinventing "how stressed is the human?" or "should the bot calm down right now?", EDON publishes a single semantic state object the stack can subscribe to.

## HumanoidEmotionState.msg

EDON exposes the robot-facing state as a custom ROS 2 message:

**Topic:** `/humanoid/emotion_state`  
**Type:** `humanoid_edon_bridge/msg/HumanoidEmotionState`

### Fields

- `string edon_mode` - EDON's primary mode: `"overload"`, `"balanced"`, `"focus"`, `"restorative"`, or `"unknown"`
- `float32 state_confidence` - 0.0–1.0 model confidence in this state
- `float32 stress_probability` - Probability of stress (from EDON parts.p_stress)
- `string locomotion_profile` - e.g. `"slow"`, `"normal"`, `"fast"`
- `string voice_profile` - e.g. `"soft"`, `"neutral"`, `"crisp"`
- `string gaze_profile` - e.g. `"soft_follow"`, `"direct"`
- `string environment_profile` - e.g. `"dim_lights"`, `"normal_lights"`
- `float32 bio_score` - Optional EDON component scores (0–1)
- `float32 env_score` - Optional EDON component scores (0–1)
- `float32 circadian_score` - Optional EDON component scores (0–1)
- `builtin_interfaces/Time stamp` - Timestamp when EDON computed this state

This is the one packet humanoid stacks need: what state the human is in, how confident EDON is, and what behavior profiles the robot should adopt.

## Mode → Behavior Profiles

EDON provides a stable mapping from internal state to behavior profiles:

| edon_mode | locomotion_profile | voice_profile | gaze_profile | environment_profile |
|-----------|-------------------|---------------|--------------|---------------------|
| `overload` | `slow` | `soft` | `soft_follow` | `dim_lights` |
| `balanced` | `normal` | `neutral` | `direct` | `normal_lights` |
| `focus` | `fast` | `crisp` | `direct` | `normal_lights` |
| `restorative` | `slow` | `soft` | `soft_follow` | `dim_lights` |
| `unknown` | `normal` | `neutral` | `direct` | `normal_lights` |

Each robot team maps these semantic labels to their own low-level commands:

- **locomotion_profile** → gait speed / max velocity / acceleration
- **voice_profile** → TTS style, volume, prosody
- **gaze_profile** → head/eye target behavior
- **environment_profile** → room lights, ambient effects, etc.

## Quick Integration (for Robot Teams)

### 1. Run EDON and the bridge

EDON is exposed as an HTTP API. The ROS 2 bridge polls `/_debug/state` and publishes `HumanoidEmotionState`.

**Environment variables:**

```bash
export EDON_API_BASE=http://<edon-host>:8001
export EDON_AUTH_ENABLED=true
export EDON_API_TOKEN=dev-token
export EDON_HUMANOID_POLL_SEC=3.0
```

**Build the package:**

```bash
cd ros2/humanoid_edon_bridge
colcon build --packages-select humanoid_edon_bridge
source install/setup.bash
```

**Run the bridge:**

```bash
ros2 run humanoid_edon_bridge edon_humanoid_bridge
```

### 2. Subscribe to `/humanoid/emotion_state`

**Minimal example controller:**

```python
from humanoid_edon_bridge.msg import HumanoidEmotionState
import rclpy
from rclpy.node import Node

class EmotionAwareController(Node):
    def __init__(self):
        super().__init__("emotion_aware_controller")
        self.sub = self.create_subscription(
            HumanoidEmotionState,
            "/humanoid/emotion_state",
            self.callback,
            10,
        )

    def callback(self, msg: HumanoidEmotionState):
        # Example logic: overload → gentle mode
        if msg.edon_mode == "overload" or msg.stress_probability > 0.7:
            self.get_logger().info("User overloaded → gentle mode.")
            # slow down locomotion, soften voice, soften gaze, dim lights, etc.
        elif msg.edon_mode == "focus":
            self.get_logger().info("User focused → high-performance mode.")
            # crisp voice, faster locomotion, direct gaze, etc.
        else:
            # balanced / restorative / unknown
            self.get_logger().info(f"EDON mode={msg.edon_mode}, staying in default behavior.")
```

From here, locomotion, voice, gaze, and environment nodes can either:

- Subscribe directly to `/humanoid/emotion_state`, or
- You can fan out from this controller into your own internal topics.

## Environment Variables

- `EDON_API_BASE` (default: `http://127.0.0.1:8001`) - EDON API base URL
- `EDON_API_TOKEN` - API token for authentication (if enabled)
- `EDON_AUTH_ENABLED` (default: `false`) - Enable bearer token authentication
- `EDON_HUMANOID_POLL_SEC` (default: `3.0`) - Polling interval in seconds

## Prerequisites

- ROS 2 (Humble, Iron, or later)
- Python 3.11+
- `requests` Python package

## License

Apache-2.0 - EDON Labs
