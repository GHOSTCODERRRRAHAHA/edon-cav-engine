import math
import time
from typing import Dict, Tuple

from edon import EdonClient, TransportType


WINDOW_LEN = 240  # 4s @ 60Hz — matches engine expectations


def make_synthetic_window(t: float, stress_mode: bool = False) -> Dict:
    """
    Fake sensor window simulating a humanoid's physiological and environmental inputs.
    
    This version forces high p_stress when stress_mode=True.
    """
    
    if stress_mode:
        # Force extreme EDA spikes + motion noise
        eda = [2.0 + 0.8 * math.sin(0.4 * k + t) for k in range(WINDOW_LEN)]
        acc_x = [0.3 * math.sin(0.4 * k) for k in range(WINDOW_LEN)]
        acc_y = [0.3 * math.cos(0.4 * k) for k in range(WINDOW_LEN)]
    else:
        eda = [0.1 + 0.01 * math.sin(0.1 * k + t) for k in range(WINDOW_LEN)]
        acc_x = [0.01 * math.sin(0.1 * k) for k in range(WINDOW_LEN)]
        acc_y = [0.01 * math.cos(0.1 * k) for k in range(WINDOW_LEN)]
    
    bvp = [math.sin((k + t * 10.0) / 8.0) for k in range(WINDOW_LEN)]
    acc_z = [1.0 for _ in range(WINDOW_LEN)]
    
    # Environmental spikes during stress
    temp_c = 22.0 if not stress_mode else 35.0
    humidity = 45.0 if not stress_mode else 70.0
    aqi = 20 if not stress_mode else 180  # Worst-case air quality
    
    return {
        "EDA": eda,
        "TEMP": [36.5 for _ in range(WINDOW_LEN)],
        "BVP": bvp,
        "ACC_x": acc_x,
        "ACC_y": acc_y,
        "ACC_z": acc_z,
        "temp_c": temp_c,
        "humidity": humidity,
        "aqi": aqi,
        "local_hour": 14,
    }


def state_to_controls(state: str, p_stress: float) -> Tuple[float, float, float]:
    """
    Convert EDON state into control envelope coefficients.
    These values would be applied to joint commands by an OEM.
    """
    if state == "restorative":
        speed = 0.7
        torque = 0.7
        safety = 0.90
    elif state == "balanced":
        speed = 1.0
        torque = 1.0
        safety = 0.85
    elif state == "focus":
        speed = 1.1
        torque = 1.1
        safety = 0.75
    else:  # overload
        speed = 0.5
        torque = 0.6
        safety = 0.95

    # Tweak safety based on p_stress
    safety = min(0.99, max(0.5, safety + 0.1 * (p_stress - 0.5)))
    return speed, torque, safety


def run_fake_robot(transport: str = "rest", steps: int = 40) -> None:
    if transport.lower() == "grpc":
        client = EdonClient(transport=TransportType.GRPC)
        transport_label = "gRPC"
    else:
        client = EdonClient()
        transport_label = "REST"

    print("\n============================")
    print(" EDON Fake Humanoid Client")
    print("============================")
    print(f" Transport: {transport_label}")
    print(f" Steps:     {steps}\n")

    # Only works for REST
    try:
        print("[EDON] Health:", client.health(), "\n")
    except Exception:
        pass

    t = 0.0
    dt = 0.2

    for step in range(steps):
        stress_mode = (step >= steps // 2)

        window = make_synthetic_window(t, stress_mode=stress_mode)
        result = client.cav(window)

        state = result.get("state")
        cav_smooth = result.get("cav_smooth")
        p_stress = result["parts"]["p_stress"]

        speed, torque, safety = state_to_controls(state, p_stress)

        print(f"[STEP {step:02d}] stress_mode={stress_mode}")
        print(f"  EDON → state={state:11s} | cav={cav_smooth} | p_stress={p_stress:.3f}")
        print(f"  Controls → speed={speed:.2f} | torque={torque:.2f} | safety={safety:.2f}")
        print("-" * 60)

        t += dt
        time.sleep(0.2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fake OEM humanoid client for EDON.")
    parser.add_argument("--transport", choices=["rest", "grpc"], default="rest")
    parser.add_argument("--steps", type=int, default=40)

    args = parser.parse_args()
    run_fake_robot(args.transport, args.steps)
