import os
import time
from typing import Any, Dict, Optional
import requests

BASE = os.getenv("EDON_API_BASE", "http://127.0.0.1:8001")
TOKEN = os.getenv("EDON_API_TOKEN")
AUTH_ENABLED = os.getenv("EDON_AUTH_ENABLED", "false").lower() == "true"
POLL_INTERVAL = float(os.getenv("EDON_HUMANOID_POLL_SEC", "3.0"))

def get_headers() -> Dict[str, str]:
    headers = {}
    if AUTH_ENABLED and TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    return headers

def fetch_state() -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(f"{BASE}/_debug/state", headers=get_headers(), timeout=3)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[HUMANOID] WARN: failed to fetch state: {e}")
        return None

def derive_mode(state: Dict[str, Any]) -> str:
    """
    Infer a high-level mode ('overload', 'balanced', 'focus', 'restorative')
    from the _debug/state payload, no matter how it's nested.
    """

    VALID = {"overload", "balanced", "focus", "restorative"}

    # 1) Try top-level string fields
    for key in ("mode", "state", "state_label", "label"):
        val = state.get(key)
        if isinstance(val, str) and val.lower() in VALID:
            return val.lower()

    # 2) Try nested 'state' / 'last_state' blocks
    for container_key in ("state", "last_state", "current", "latest"):
        container = state.get(container_key)
        if isinstance(container, dict):
            # Look for explicit string fields
            for key in ("mode", "state", "state_label", "label"):
                val = container.get(key)
                if isinstance(val, str) and val.lower() in VALID:
                    return val.lower()

            # Look for any value that matches a known mode
            for v in container.values():
                if isinstance(v, str) and v.lower() in VALID:
                    return v.lower()

            # Numeric mapping inside container
            numeric = container.get("cav_state") or container.get("class_id")
            if isinstance(numeric, int):
                return {
                    1: "overload",
                    2: "balanced",
                    3: "focus",
                    4: "restorative",
                }.get(numeric, "unknown")

    # 3) Fallback numeric mapping at top level
    numeric = state.get("cav_state") or state.get("class_id")
    if isinstance(numeric, int):
        return {
            1: "overload",
            2: "balanced",
            3: "focus",
            4: "restorative",
        }.get(numeric, "unknown")

    # 4) As a last resort, scan all strings in the payload
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

def decide_actions(state: Dict[str, Any]) -> Dict[str, Any]:
    mode = derive_mode(state)
    if mode == "overload":
        return {
            "mode": mode,
            "locomotion_speed": "slow",
            "voice_tone": "soft",
            "gaze_mode": "soft_follow",
            "environment": "dim_lights",
        }
    if mode == "balanced":
        return {
            "mode": mode,
            "locomotion_speed": "normal",
            "voice_tone": "neutral",
            "gaze_mode": "direct",
            "environment": "normal_lights",
        }
    if mode == "focus":
        return {
            "mode": mode,
            "locomotion_speed": "fast",
            "voice_tone": "crisp",
            "gaze_mode": "direct",
            "environment": "normal_lights",
        }
    if mode == "restorative":
        return {
            "mode": mode,
            "locomotion_speed": "slow",
            "voice_tone": "soft",
            "gaze_mode": "soft_follow",
            "environment": "dim_lights",
        }
    return {
        "mode": "unknown",
        "locomotion_speed": "normal",
        "voice_tone": "neutral",
        "gaze_mode": "direct",
        "environment": "normal_lights",
    }

def main():
    print(f"[HUMANOID] Starting consumer. Base={BASE} interval={POLL_INTERVAL}s")
    while True:
        state = fetch_state()
        if state:
            # TEMP DEBUG: print the raw state once
            if os.getenv("EDON_HUMANOID_DEBUG", "0") == "1":
                from pprint import pprint
                print("[HUMANOID DEBUG] raw _debug/state payload:")
                pprint(state)
                # After inspecting once, you can turn this env off.

            actions = decide_actions(state)
            print(
                f"[HUMANOID] mode={actions['mode']} "
                f"speed={actions['locomotion_speed']} "
                f"voice={actions['voice_tone']} "
                f"gaze={actions['gaze_mode']} "
                f"env={actions['environment']}"
            )
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[HUMANOID] Stopped.")
