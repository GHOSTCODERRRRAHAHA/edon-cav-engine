import os
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
import requests

from humanoid_edon_bridge.msg import HumanoidEmotionState


def derive_mode(state: Dict[str, Any]) -> str:
    """Infer EDON mode from _debug/state payload."""
    VALID = {"overload", "balanced", "focus", "restorative"}

    # 1) Top-level
    for key in ("mode", "state", "state_label", "label"):
        val = state.get(key)
        if isinstance(val, str) and val.lower() in VALID:
            return val.lower()

    # 2) Nested containers
    for container_key in ("state", "last_state", "current", "latest"):
        container = state.get(container_key)
        if isinstance(container, dict):
            for key in ("mode", "state", "state_label", "label"):
                val = container.get(key)
                if isinstance(val, str) and val.lower() in VALID:
                    return val.lower()

            numeric = container.get("cav_state") or container.get("class_id")
            if isinstance(numeric, int):
                return {
                    1: "overload",
                    2: "balanced",
                    3: "focus",
                    4: "restorative",
                }.get(numeric, "unknown")

            for v in container.values():
                if isinstance(v, str) and v.lower() in VALID:
                    return v.lower()

    # 3) Fallback numeric at top-level
    numeric = state.get("cav_state") or state.get("class_id")
    if isinstance(numeric, int):
        return {
            1: "overload",
            2: "balanced",
            3: "focus",
            4: "restorative",
        }.get(numeric, "unknown")

    # 4) Deep scan
    def scan(obj: Any) -> Optional[str]:
        if isinstance(obj, dict):
            for v in obj.values():
                found = scan(v)
                if found:
                    return found
        elif isinstance(obj, list):
            for v in obj:
                found = scan(v)
                if found:
                    return found
        elif isinstance(obj, str) and obj.lower() in VALID:
            return obj.lower()
        return None

    found = scan(state)
    if found:
        return found

    return "unknown"


def map_mode_to_profiles(mode: str) -> Dict[str, str]:
    """Map EDON mode to locomotion / voice / gaze / environment profiles."""
    if mode == "overload":
        return {
            "locomotion": "slow",
            "voice": "soft",
            "gaze": "soft_follow",
            "env": "dim_lights",
        }
    if mode == "balanced":
        return {
            "locomotion": "normal",
            "voice": "neutral",
            "gaze": "direct",
            "env": "normal_lights",
        }
    if mode == "focus":
        return {
            "locomotion": "fast",
            "voice": "crisp",
            "gaze": "direct",
            "env": "normal_lights",
        }
    if mode == "restorative":
        return {
            "locomotion": "slow",
            "voice": "soft",
            "gaze": "soft_follow",
            "env": "dim_lights",
        }

    # unknown fallback
    return {
        "locomotion": "normal",
        "voice": "neutral",
        "gaze": "direct",
        "env": "normal_lights",
    }


class EdonHumanoidBridge(Node):
    def __init__(self) -> None:
        super().__init__("edon_humanoid_bridge")

        self.base = os.getenv("EDON_API_BASE", "http://127.0.0.1:8001")
        self.token = os.getenv("EDON_API_TOKEN")
        self.auth_enabled = os.getenv("EDON_AUTH_ENABLED", "false").lower() == "true"
        poll_sec = float(os.getenv("EDON_HUMANOID_POLL_SEC", "3.0"))

        self.get_logger().info(
            f"EDON Humanoid Bridge started. Base={self.base} poll={poll_sec}s"
        )

        # Publisher for unified emotion state message
        self.emotion_state_pub = self.create_publisher(
            HumanoidEmotionState, "/humanoid/emotion_state", 10
        )

        # Timer
        self.timer = self.create_timer(poll_sec, self.timer_callback)

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.auth_enabled and self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def fetch_state(self) -> Optional[Dict[str, Any]]:
        try:
            resp = requests.get(
                f"{self.base}/_debug/state", headers=self._headers(), timeout=3.0
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"Failed to fetch EDON state: {e}")
            return None

    def timer_callback(self) -> None:
        payload = self.fetch_state()
        if not payload:
            return

        mode = derive_mode(payload)
        profiles = map_mode_to_profiles(mode)

        # Extract nested state fields (matches what you saw in debug)
        state_block = payload.get("state") or {}
        confidence = state_block.get("confidence", 0.0)
        parts = state_block.get("parts") or {}
        p_stress = parts.get("p_stress", 0.0)
        bio_score = parts.get("bio", 0.0)
        env_score = parts.get("env", 0.0)
        circadian_score = parts.get("circadian", 0.0)

        # Create unified emotion state message
        msg = HumanoidEmotionState()
        msg.edon_mode = mode
        msg.state_confidence = float(confidence)
        msg.stress_probability = float(p_stress)
        msg.locomotion_profile = profiles["locomotion"]
        msg.voice_profile = profiles["voice"]
        msg.gaze_profile = profiles["gaze"]
        msg.environment_profile = profiles["env"]
        msg.bio_score = float(bio_score)
        msg.env_score = float(env_score)
        msg.circadian_score = float(circadian_score)
        
        # Set timestamp to current time
        msg.stamp = self.get_clock().now().to_msg()

        # Publish unified message
        self.emotion_state_pub.publish(msg)

        self.get_logger().info(
            f"mode={mode} locomotion={profiles['locomotion']} "
            f"voice={profiles['voice']} gaze={profiles['gaze']} "
            f"env={profiles['env']} conf={confidence:.2f} p_stress={p_stress:.3f} "
            f"bio={bio_score:.2f} env={env_score:.2f} circ={circadian_score:.2f}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EdonHumanoidBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

