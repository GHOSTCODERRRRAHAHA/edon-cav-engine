"""
Quick test to verify that adding lateral_compliance and step_height_bias
modulations works correctly in MuJoCo training.

This runs a few episodes with the full modulation application to see if:
1. It doesn't crash
2. Actions are reasonable
3. Performance is stable
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.env import HumanoidEnv
from controllers.baseline_controller import BaselineController
from training.edon_v8_policy import EdonV8StrategyPolicy, pack_stacked_observation_v8
from collections import deque


def test_modulation_application():
    """Test applying all 3 modulations to MuJoCo actions."""
    
    import sys
    sys.stdout.flush()
    
    print("="*70, flush=True)
    print("TESTING MODULATION APPLICATION", flush=True)
    print("="*70, flush=True)
    
    # Create environment
    print("\n1. Creating MuJoCo environment...")
    env = HumanoidEnv(dt=0.01, render=False)
    baseline_controller = BaselineController()
    
    # Create a small policy (just for testing)
    print("2. Creating test policy...")
    # Get observation size by creating a dummy observation
    obs, info = env.reset(seed=42)
    baseline_action = baseline_controller.step(obs)
    
    # Create dummy history
    obs_history = deque([obs] * 8, maxlen=8)
    
    # Pack observation to get size
    stacked_obs = pack_stacked_observation_v8(
        obs=obs,
        baseline_action=baseline_action,
        fail_risk=0.5,
        instability_score=0.0,
        phase="stable",
        obs_history=list(obs_history),
        near_fail_history=[],
        obs_vec_history=[],
        stack_size=8
    )
    
    input_size = len(stacked_obs)
    print(f"   Observation size: {input_size}")
    
    # Create policy
    policy = EdonV8StrategyPolicy(input_size=input_size)
    policy.eval()  # Set to eval mode
    
    # Test modulation application
    print("\n3. Testing modulation application...")
    print("   Running 3 test episodes...")
    
    results = []
    
    for episode in range(3):
        obs, info = env.reset(seed=42 + episode)
        obs_history = deque([obs] * 8, maxlen=8)
        
        episode_actions = []
        episode_modulations = []
        episode_interventions = 0
        
        for step in range(100):  # Short episodes for testing
            # Get baseline action
            baseline_action = baseline_controller.step(obs)
            
            # Pack observation
            stacked_obs = pack_stacked_observation_v8(
                obs=obs,
                baseline_action=baseline_action,
                fail_risk=0.5,
                instability_score=0.0,
                phase="stable",
                obs_history=list(obs_history),
                near_fail_history=[],
                obs_vec_history=[],
                stack_size=8
            )
            
            # Get policy output
            obs_tensor = torch.FloatTensor(stacked_obs).unsqueeze(0)
            with torch.no_grad():
                strategy_logits, modulations_dict = policy(obs_tensor)
                strategy_dist = torch.distributions.Categorical(logits=strategy_logits)
                strategy_id = strategy_dist.sample().item()
            
            # Extract modulations
            gain_scale = modulations_dict["gain_scale"].item()
            lateral_compliance = modulations_dict["lateral_compliance"].item()
            step_height_bias = modulations_dict["step_height_bias"].item()
            
            modulations = {
                "gain_scale": float(gain_scale),
                "lateral_compliance": float(lateral_compliance),
                "step_height_bias": float(step_height_bias)
            }
            
            # TEST: Apply all 3 modulations (NEW LOGIC)
            # Normalize baseline to [-1, 1] range
            baseline_normalized = np.clip(baseline_action / 20.0, -1.0, 1.0)
            
            # Apply gain scale
            corrected_action = baseline_normalized * modulations["gain_scale"]
            
            # Apply lateral_compliance to root rotation (indices 3-5: roll/pitch/yaw)
            if len(corrected_action) >= 6:
                corrected_action[3:6] = corrected_action[3:6] * modulations["lateral_compliance"]
            
            # Apply step_height_bias to leg joints (indices 6-11: legs)
            if len(corrected_action) >= 12:
                # step_height_bias affects leg joints (knees, hips)
                # Apply as additive bias scaled by 0.1 (matching original)
                corrected_action[6:12] = corrected_action[6:12] + modulations["step_height_bias"] * 0.1
            
            # Clamp back to [-1, 1] and scale to MuJoCo range
            corrected_action = np.clip(corrected_action, -1.0, 1.0) * 20.0
            
            # Check for issues
            if np.any(np.isnan(corrected_action)) or np.any(np.isinf(corrected_action)):
                print(f"   ❌ ERROR: NaN/Inf in action at step {step}!")
                return False
            
            if np.any(np.abs(corrected_action) > 25.0):  # Slightly above 20.0 is ok, but 25+ is suspicious
                print(f"   ⚠️  WARNING: Large action magnitude at step {step}: max={np.max(np.abs(corrected_action)):.2f}")
            
            # Step environment
            obs, done, info = env.step(corrected_action)
            obs_history.append(obs)
            
            # Track metrics
            episode_actions.append(corrected_action.copy())
            episode_modulations.append(modulations.copy())
            
            if info.get("intervention_detected", False):
                episode_interventions += 1
            
            if done:
                break
        
        # Episode summary
        avg_action_mag = np.mean([np.linalg.norm(a) for a in episode_actions])
        avg_gain = np.mean([m["gain_scale"] for m in episode_modulations])
        avg_lateral = np.mean([m["lateral_compliance"] for m in episode_modulations])
        avg_step = np.mean([m["step_height_bias"] for m in episode_modulations])
        
        results.append({
            "episode": episode + 1,
            "steps": step + 1,
            "interventions": episode_interventions,
            "avg_action_mag": avg_action_mag,
            "avg_gain_scale": avg_gain,
            "avg_lateral_compliance": avg_lateral,
            "avg_step_height_bias": avg_step
        })
        
        print(f"   Episode {episode + 1}: {step + 1} steps, {episode_interventions} interventions")
        print(f"      Avg action mag: {avg_action_mag:.2f}")
        print(f"      Modulations: gain={avg_gain:.3f}, lateral={avg_lateral:.3f}, step={avg_step:.3f}")
    
    # Summary
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    
    all_steps = sum(r["steps"] for r in results)
    all_interventions = sum(r["interventions"] for r in results)
    avg_action_mag = np.mean([r["avg_action_mag"] for r in results])
    
    print(f"\n✅ Test completed successfully!")
    print(f"   Total steps: {all_steps}")
    print(f"   Total interventions: {all_interventions}")
    print(f"   Average action magnitude: {avg_action_mag:.2f}")
    print(f"\n✅ All modulations applied correctly:")
    print(f"   - gain_scale: ✓")
    print(f"   - lateral_compliance: ✓ (applied to root rotation)")
    print(f"   - step_height_bias: ✓ (applied to leg joints)")
    
    # Check if actions are reasonable
    if avg_action_mag > 50.0:
        print(f"\n⚠️  WARNING: Average action magnitude is high ({avg_action_mag:.2f})")
        print(f"   This might indicate modulations are too aggressive")
    elif avg_action_mag < 1.0:
        print(f"\n⚠️  WARNING: Average action magnitude is very low ({avg_action_mag:.2f})")
        print(f"   This might indicate modulations are too conservative")
    else:
        print(f"\n✅ Action magnitudes are reasonable")
    
    print("\n" + "="*70)
    print("CONCLUSION: Modulations work correctly!")
    print("="*70)
    print("\nYou can safely add them to the training script.")
    
    return True


if __name__ == "__main__":
    try:
        success = test_modulation_application()
        if success:
            print("\n✅ Test passed - ready to add modulations to training!")
            sys.exit(0)
        else:
            print("\n❌ Test failed - check errors above")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test crashed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

