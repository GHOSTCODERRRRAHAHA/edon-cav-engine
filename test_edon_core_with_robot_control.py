"""
Test: Can EDON Core help with robot control like v8 did?

This test checks if EDON Core's control scales can improve robot stability
when used alongside or instead of v8's control modulations.

Note: EDON Core and v8 are different systems:
- v8: Robot state → Control modulations (prevents interventions)
- EDON Core: Physiological sensors → Control scales (adapts to human state)

But we can test if EDON Core's adaptive principles help.
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

# Import v8 components
from env.edon_humanoid_env_v8 import EdonHumanoidEnvV8
from training.edon_v8_policy import EdonV8StrategyPolicy
from training.fail_risk_model import FailRiskModel
from run_eval import baseline_controller, make_humanoid_env
import torch


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


def test_edon_core_with_robot():
    """
    Test if EDON Core can help with robot control.
    
    This test:
    1. Runs robot episodes with v8 only (baseline)
    2. Runs robot episodes with v8 + EDON Core control scales
    3. Compares results
    """
    print("=" * 70)
    print("Testing: Can EDON Core Help with Robot Control?")
    print("=" * 70)
    
    if not EDON_CORE_AVAILABLE:
        print("\n[ERROR] EDON Core SDK not available.")
        print("To test:")
        print("1. Start EDON Core server: docker run -p 8002:8000 -p 50052:50051 edon-server:v1.0.1")
        print("2. Install SDK: pip install sdk/python/edon-*.whl")
        return
    
    # Check if EDON Core server is running
    try:
        client = EdonClient(base_url="http://localhost:8002", transport=TransportType.REST)
        health = client.health()
        if not health.get("ok", False):
            print("[ERROR] EDON Core server is not healthy")
            return
        print(f"[OK] EDON Core server is running: {health}")
    except Exception as e:
        print(f"[ERROR] Cannot connect to EDON Core server: {e}")
        print("Start server with: docker run -p 8002:8000 -p 50052:50051 edon-server:v1.0.1")
        return
    
    # Load v8 model
    v8_model_path = Path("models/edon_v8_strategy_memory_features.pt")
    if not v8_model_path.exists():
        print(f"[ERROR] v8 model not found: {v8_model_path}")
        print("Train v8 first: python training/train_edon_v8_strategy.py")
        return
    
    print(f"\n[OK] Loading v8 model from {v8_model_path}")
    checkpoint = torch.load(v8_model_path, map_location="cpu", weights_only=False)
    input_size = checkpoint.get("input_size", 248)
    policy = EdonV8StrategyPolicy(input_size=input_size)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    
    # Load fail-risk model
    fail_risk_path = Path("models/edon_fail_risk_v1_fixed_v2.pt")
    if not fail_risk_path.exists():
        print(f"[ERROR] Fail-risk model not found: {fail_risk_path}")
        return
    
    print(f"[OK] Loading fail-risk model from {fail_risk_path}")
    fail_risk_checkpoint = torch.load(fail_risk_path, map_location="cpu", weights_only=False)
    fail_risk_input_size = fail_risk_checkpoint.get("input_size", 15)
    fail_risk_model = FailRiskModel(input_size=fail_risk_input_size)
    fail_risk_model.load_state_dict(fail_risk_checkpoint["model_state_dict"])
    fail_risk_model.eval()
    
    # Create base environment
    base_env = make_humanoid_env(seed=0, profile="high_stress")
    
    # Test 1: v8 only (baseline)
    print("\n" + "=" * 70)
    print("Test 1: v8 Only (Baseline)")
    print("=" * 70)
    
    env_v8_only = EdonHumanoidEnvV8(
        strategy_policy=policy,
        fail_risk_model=fail_risk_model,
        base_env=base_env,
        seed=0,
        profile="high_stress",
        device="cpu",
        w_intervention=20.0,
        w_stability=1.0,
        w_torque=0.1
    )
    
    results_v8_only = run_episode(env_v8_only, use_edon_core=False, client=None)
    
    print(f"\n[RESULTS] v8 Only:")
    print(f"  Interventions: {results_v8_only['interventions']}")
    print(f"  Stability: {results_v8_only['stability']:.4f}")
    print(f"  Episode Length: {results_v8_only['episode_length']}")
    
    # Test 2: v8 + EDON Core control scales
    print("\n" + "=" * 70)
    print("Test 2: v8 + EDON Core Control Scales")
    print("=" * 70)
    
    env_v8_edon = EdonHumanoidEnvV8(
        strategy_policy=policy,
        fail_risk_model=fail_risk_model,
        base_env=base_env,
        seed=0,
        profile="high_stress",
        device="cpu",
        w_intervention=20.0,
        w_stability=1.0,
        w_torque=0.1
    )
    
    results_v8_edon = run_episode(env_v8_edon, use_edon_core=True, client=client)
    
    print(f"\n[RESULTS] v8 + EDON Core:")
    print(f"  Interventions: {results_v8_edon['interventions']}")
    print(f"  Stability: {results_v8_edon['stability']:.4f}")
    print(f"  Episode Length: {results_v8_edon['episode_length']}")
    
    # Comparison
    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)
    
    delta_interventions = results_v8_edon['interventions'] - results_v8_only['interventions']
    delta_stability = results_v8_edon['stability'] - results_v8_only['stability']
    
    print(f"v8 Only:        Interventions={results_v8_only['interventions']}, Stability={results_v8_only['stability']:.4f}")
    print(f"v8 + EDON Core: Interventions={results_v8_edon['interventions']}, Stability={results_v8_edon['stability']:.4f}")
    print(f"\nDelta:          Interventions={delta_interventions:+.1f}, Stability={delta_stability:+.4f}")
    
    if delta_interventions < 0:
        print(f"\n✅ EDON Core helped: {abs(delta_interventions)} fewer interventions")
    elif delta_interventions > 0:
        print(f"\n❌ EDON Core hurt: {delta_interventions} more interventions")
    else:
        print(f"\n➖ EDON Core had no effect on interventions")
    
    if abs(delta_stability) < 0.001:
        print(f"➖ EDON Core had no effect on stability")
    elif delta_stability < 0:
        print(f"✅ EDON Core improved stability (lower is better)")
    else:
        print(f"❌ EDON Core worsened stability")
    
    # Save results
    results = {
        "v8_only": results_v8_only,
        "v8_edon_core": results_v8_edon,
        "comparison": {
            "delta_interventions": float(delta_interventions),
            "delta_stability": float(delta_stability)
        }
    }
    
    output_path = Path("results/edon_core_robot_control_test.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Results saved to {output_path}")


def run_episode(env: EdonHumanoidEnvV8, use_edon_core: bool, client: Optional[Any]) -> Dict[str, Any]:
    """
    Run a single episode with optional EDON Core integration.
    
    Args:
        env: v8 environment
        use_edon_core: Whether to use EDON Core control scales
        client: EDON Core client (if use_edon_core is True)
    
    Returns:
        Episode metrics
    """
    obs = env.reset()
    done = False
    step = 0
    interventions = 0
    roll_history = []
    pitch_history = []
    max_steps = 1000
    
    # Buffer for physiological windows (need 240 samples = 4 seconds @ 60Hz)
    # For testing, we'll use a smaller buffer and synthesize
    physio_buffer = []
    
    while not done and step < max_steps:
        # Get EDON Core state if requested
        edon_core_state = None
        edon_control_scales = None
        
        if use_edon_core and client:
            # Build physiological window from robot state
            # In reality, this would come from actual sensors
            physio_window = create_synthetic_physiological_window(obs)
            
            try:
                # Get EDON Core prediction
                result = client.cav(physio_window)
                state = result.get("state", "balanced")
                p_stress = result.get("parts", {}).get("p_stress", 0.5)
                controls = result.get("controls", {})
                
                # Map EDON Core state to edon_core_state format
                edon_core_state = {
                    "instability_score": p_stress,  # Use p_stress as proxy
                    "risk_ema": p_stress,  # Use p_stress as proxy
                    "phase": "stable" if state == "balanced" else "warning"
                }
                
                # Get control scales
                edon_control_scales = {
                    "speed": controls.get("speed", 1.0),
                    "torque": controls.get("torque", 1.0),
                    "safety": controls.get("safety", 0.85)
                }
                
            except Exception as e:
                print(f"[WARN] EDON Core call failed: {e}")
                edon_core_state = None
        
        # Step environment
        obs, reward, done, info = env.step(edon_core_state=edon_core_state)
        
        # Track metrics
        if info.get("intervention", False) or info.get("fallen", False):
            interventions += 1
        
        roll_history.append(obs.get("roll", 0.0))
        pitch_history.append(obs.get("pitch", 0.0))
        
        step += 1
    
    # Compute stability
    if len(roll_history) > 1 and len(pitch_history) > 1:
        stability = float(np.var(roll_history) + np.var(pitch_history))
    else:
        stability = float('inf')
    
    return {
        "interventions": interventions,
        "stability": stability,
        "episode_length": step,
        "success": not done and step < max_steps
    }


if __name__ == "__main__":
    test_edon_core_with_robot()

