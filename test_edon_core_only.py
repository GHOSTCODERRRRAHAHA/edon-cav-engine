"""
Test: EDON Core Only (No v8)

This test checks if EDON Core's control scales can improve robot stability
when applied directly to baseline controller actions, without v8.

Test setup:
- Baseline: baseline_controller only
- EDON Core Only: baseline_controller + EDON Core control scales
"""

import numpy as np
from pathlib import Path
import json
from typing import Dict, Any, Optional

# Try to import EDON Core
try:
    from edon import EdonClient, TransportType
    EDON_CORE_AVAILABLE = True
except ImportError:
    try:
        from sdk.python.edon.client import EdonClient, TransportType
        EDON_CORE_AVAILABLE = True
    except ImportError:
        print("[WARN] EDON Core SDK not available. Install with: pip install sdk/python/edon-*.whl")
        EDON_CORE_AVAILABLE = False
        EdonClient = None
        TransportType = None

# Import components
from run_eval import baseline_controller, make_humanoid_env
from metrics.edon_v8_metrics import compute_episode_metrics_v8


def create_synthetic_physiological_window(robot_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create synthetic physiological sensor window from robot state.
    
    This is a test - in reality, EDON Core would get real physiological sensors.
    For testing, we'll map robot state to synthetic physiological signals.
    
    Args:
        robot_state: Robot observation dict with roll, pitch, velocities, etc.
    
    Returns:
        Physiological sensor window (240 samples) for EDON Core
    """
    roll = robot_state.get("roll", 0.0)
    pitch = robot_state.get("pitch", 0.0)
    roll_vel = robot_state.get("roll_velocity", 0.0)
    pitch_vel = robot_state.get("pitch_velocity", 0.0)
    
    # Map robot instability to synthetic stress signals
    # High tilt → high stress (synthetic mapping for testing)
    tilt_mag = np.sqrt(roll**2 + pitch**2)
    vel_mag = np.sqrt(roll_vel**2 + pitch_vel**2)
    
    # Synthetic EDA (electrodermal activity) - higher when robot is unstable
    base_eda = 0.1
    stress_eda = min(2.0, base_eda + tilt_mag * 5.0 + vel_mag * 2.0)
    eda_signal = np.random.normal(stress_eda, 0.05, 240).clip(0.01, 2.0)
    
    # Synthetic BVP (blood volume pulse) - higher when robot is unstable
    base_bvp = 0.5
    stress_bvp = min(1.0, base_bvp + tilt_mag * 2.0)
    bvp_signal = np.random.normal(stress_bvp, 0.1, 240).clip(0.0, 1.0)
    
    # Synthetic accelerometer (from robot motion)
    acc_x = np.random.normal(roll_vel * 0.1, 0.05, 240)
    acc_y = np.random.normal(pitch_vel * 0.1, 0.05, 240)
    acc_z = np.random.normal(1.0, 0.05, 240)  # Gravity
    
    # Temperature (normal)
    temp_signal = np.full(240, 36.5)
    
    return {
        "EDA": eda_signal.tolist(),
        "TEMP": temp_signal.tolist(),
        "BVP": bvp_signal.tolist(),
        "ACC_x": acc_x.tolist(),
        "ACC_y": acc_y.tolist(),
        "ACC_z": acc_z.tolist(),
        "temp_c": 22.0,
        "humidity": 50.0,
        "aqi": 35,
        "local_hour": 14
    }


def apply_edon_core_scales(action: np.ndarray, scales: Dict[str, float]) -> np.ndarray:
    """
    Apply EDON Core control scales to an action.
    
    Args:
        action: Baseline action array
        scales: Dict with 'speed', 'torque', 'safety' scales
    
    Returns:
        Scaled action array
    """
    scaled_action = action.copy()
    
    # Apply speed scale to forward/backward movements (indices 4-8 as example)
    # Adjust based on your action space
    if len(scaled_action) >= 8:
        scaled_action[4:8] *= scales.get("speed", 1.0)
    
    # Apply torque scale to joint torques (indices 0-4 as example)
    if len(scaled_action) >= 4:
        scaled_action[:4] *= scales.get("torque", 1.0)
    
    # Apply safety scale to entire action (conservative scaling)
    scaled_action *= scales.get("safety", 1.0)
    
    # Clip to valid range
    scaled_action = np.clip(scaled_action, -1.0, 1.0)
    
    return scaled_action


def run_episode_baseline_only(env, max_steps: int = 1000) -> Dict[str, Any]:
    """Run one episode with baseline controller only."""
    obs = env.reset()
    episode_data = []
    step_count = 0
    
    while step_count < max_steps:
        # Get baseline action
        action = baseline_controller(obs, edon_state=None)
        action = np.array(action)
        
        # Step environment
        next_obs, reward, done, info = env.step(action)
        
        episode_data.append({
            "obs": obs,
            "action": action.tolist(),
            "reward": reward,
            "done": done,
            "info": info,
            "intervention": info.get("intervention", False) or info.get("fallen", False)
        })
        
        obs = next_obs
        step_count += 1
        
        if done:
            break
    
    return compute_episode_metrics_v8(episode_data)


def run_episode_edon_core_only(env, edon_client: EdonClient, max_steps: int = 1000) -> Dict[str, Any]:
    """Run one episode with baseline controller + EDON Core control scales."""
    obs = env.reset()
    episode_data = []
    step_count = 0
    
    while step_count < max_steps:
        # Get baseline action
        baseline_action = baseline_controller(obs, edon_state=None)
        baseline_action = np.array(baseline_action)
        
        # Get EDON Core control scales
        try:
            physio_window = create_synthetic_physiological_window(obs)
            cav_result = edon_client.cav(physio_window)
            control_scales = cav_result.get("controls", {"speed": 1.0, "torque": 1.0, "safety": 1.0})
        except Exception as e:
            print(f"[WARN] Could not get EDON Core scales: {e}, using defaults")
            control_scales = {"speed": 1.0, "torque": 1.0, "safety": 1.0}
        
        # Apply EDON Core scales to baseline action
        action = apply_edon_core_scales(baseline_action, control_scales)
        
        # Step environment
        next_obs, reward, done, info = env.step(action)
        
        episode_data.append({
            "obs": obs,
            "action": action.tolist(),
            "baseline_action": baseline_action.tolist(),
            "edon_scales": control_scales,
            "reward": reward,
            "done": done,
            "info": info,
            "intervention": info.get("intervention", False) or info.get("fallen", False)
        })
        
        obs = next_obs
        step_count += 1
        
        if done:
            break
    
    return compute_episode_metrics_v8(episode_data)


def test_edon_core_only():
    """
    Test EDON Core only (no v8).
    
    Compares:
    - Baseline only
    - Baseline + EDON Core control scales
    """
    print("\n" + "=" * 70)
    print("Testing: EDON Core Only (No v8)")
    print("=" * 70)
    
    # Check EDON Core availability
    if not EDON_CORE_AVAILABLE:
        print("[ERROR] EDON Core SDK not available. Cannot run test.")
        return
    
    # Check EDON Core server
    try:
        client = EdonClient(base_url="http://127.0.0.1:8002", transport=TransportType.REST, timeout=5.0)
        health = client.health()
        if not health.get("ok"):
            print(f"[ERROR] EDON Core server not healthy: {health}")
            return
        print(f"\n[OK] EDON Core server is running: {health}")
    except Exception as e:
        print(f"[ERROR] Could not connect to EDON Core server: {e}")
        print("[INFO] Make sure EDON Core server is running: .\\start_edon_core_server.ps1")
        return
    
    # Create base environment
    base_env = make_humanoid_env(seed=0, profile="high_stress")
    
    # Test configuration
    num_episodes = 10  # Quick test
    max_steps = 1000
    
    print(f"\n[CONFIG] Episodes: {num_episodes}")
    print(f"[CONFIG] Max steps per episode: {max_steps}")
    print(f"[CONFIG] Seed: 0, Profile: high_stress")
    
    # Test 1: Baseline only
    print("\n" + "=" * 70)
    print("Test 1: Baseline Only")
    print("=" * 70)
    
    baseline_results = []
    for i in range(num_episodes):
        metrics = run_episode_baseline_only(base_env, max_steps=max_steps)
        baseline_results.append(metrics)
        print(f"  Episode {i+1}/{num_episodes}: Interventions={metrics['interventions']}, Stability={metrics['stability_avg']:.4f}")
    
    baseline_avg = {
        "interventions": np.mean([r["interventions"] for r in baseline_results]),
        "stability": np.mean([r["stability_avg"] for r in baseline_results]),
        "episode_length": np.mean([r["episode_length"] for r in baseline_results]),
    }
    
    print(f"\n[RESULTS] Baseline Only:")
    print(f"  Interventions/episode: {baseline_avg['interventions']:.2f}")
    print(f"  Stability (avg): {baseline_avg['stability']:.4f}")
    print(f"  Episode length (avg): {baseline_avg['episode_length']:.1f}")
    
    # Test 2: Baseline + EDON Core
    print("\n" + "=" * 70)
    print("Test 2: Baseline + EDON Core Control Scales")
    print("=" * 70)
    
    edon_core_results = []
    for i in range(num_episodes):
        metrics = run_episode_edon_core_only(base_env, client, max_steps=max_steps)
        edon_core_results.append(metrics)
        print(f"  Episode {i+1}/{num_episodes}: Interventions={metrics['interventions']}, Stability={metrics['stability_avg']:.4f}")
    
    edon_core_avg = {
        "interventions": np.mean([r["interventions"] for r in edon_core_results]),
        "stability": np.mean([r["stability_avg"] for r in edon_core_results]),
        "episode_length": np.mean([r["episode_length"] for r in edon_core_results]),
    }
    
    print(f"\n[RESULTS] Baseline + EDON Core:")
    print(f"  Interventions/episode: {edon_core_avg['interventions']:.2f}")
    print(f"  Stability (avg): {edon_core_avg['stability']:.4f}")
    print(f"  Episode length (avg): {edon_core_avg['episode_length']:.1f}")
    
    # Comparison
    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)
    print(f"Baseline Only:        Interventions={baseline_avg['interventions']:.2f}, Stability={baseline_avg['stability']:.4f}")
    print(f"Baseline + EDON Core: Interventions={edon_core_avg['interventions']:.2f}, Stability={edon_core_avg['stability']:.4f}")
    
    delta_interventions = baseline_avg['interventions'] - edon_core_avg['interventions']
    delta_stability = baseline_avg['stability'] - edon_core_avg['stability']
    
    print(f"\nDelta:                Interventions={delta_interventions:+.2f}, Stability={delta_stability:+.4f}")
    
    if delta_interventions > 0:
        print(f"✅ EDON Core reduced interventions by {delta_interventions:.2f}")
    elif delta_interventions < 0:
        print(f"❌ EDON Core increased interventions by {abs(delta_interventions):.2f}")
    else:
        print(f"➖ EDON Core had no effect on interventions")
    
    # Note: Lower stability is better (less variance)
    if abs(delta_stability) < 0.0001:
        print(f"➖ EDON Core had no effect on stability")
    elif delta_stability > 0:
        # Positive delta means baseline was worse (higher), so EDON Core improved it
        print(f"✅ EDON Core improved stability by {delta_stability:.4f} (lower is better)")
    else:
        # Negative delta means baseline was better (lower), so EDON Core worsened it
        print(f"❌ EDON Core worsened stability by {abs(delta_stability):.4f} (lower is better)")
    
    # Save results
    results = {
        "test": "edon_core_only",
        "baseline": baseline_avg,
        "edon_core": edon_core_avg,
        "delta": {
            "interventions": delta_interventions,
            "stability": delta_stability
        },
        "episodes": num_episodes,
        "baseline_episodes": baseline_results,
        "edon_core_episodes": edon_core_results
    }
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / "edon_core_only_test.json"
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Results saved to {results_path}")


if __name__ == "__main__":
    test_edon_core_only()

