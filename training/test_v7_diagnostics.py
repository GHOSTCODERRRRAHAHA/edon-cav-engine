"""
Test script to diagnose v7 training issues.

Compares:
1. Random policy behavior
2. Trained v7 policy behavior
3. Reward statistics
4. Action delta statistics
"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.edon_humanoid_env import EdonHumanoidEnv
from run_eval import baseline_controller
from training.train_edon_v7 import EdonV7Policy, pack_observation


def test_policy(policy, policy_name, num_episodes=5):
    """Test a policy and collect statistics."""
    env = EdonHumanoidEnv(seed=42, profile="high_stress")
    
    all_rewards = []
    all_deltas = []
    all_actions = []
    episode_lengths = []
    
    print(f"\n{'='*70}")
    print(f"Testing {policy_name}")
    print(f"{'='*70}")
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_rewards = []
        episode_deltas = []
        episode_actions = []
        done = False
        step = 0
        
        while not done and step < 1000:
            baseline_action = baseline_controller(obs, edon_state=None)
            baseline_action = np.array(baseline_action)
            
            obs_vec = pack_observation(obs, baseline_action)
            obs_tensor = torch.FloatTensor(obs_vec).unsqueeze(0)
            
            with torch.no_grad():
                action_delta = policy(obs_tensor).cpu().numpy()[0]
            
            final_action = np.clip(baseline_action + action_delta, -1.0, 1.0)
            
            next_obs, reward, done, info = env.step(final_action)
            
            episode_rewards.append(reward)
            episode_deltas.append(action_delta)
            episode_actions.append(final_action)
            
            obs = next_obs
            step += 1
        
        all_rewards.extend(episode_rewards)
        all_deltas.extend(episode_deltas)
        all_actions.extend(episode_actions)
        episode_lengths.append(step)
    
    # Compute statistics
    rewards_array = np.array(all_rewards)
    deltas_array = np.array(all_deltas)
    actions_array = np.array(all_actions)
    
    print(f"\nEpisode Lengths: mean={np.mean(episode_lengths):.1f}, std={np.std(episode_lengths):.1f}")
    print(f"  Range: [{np.min(episode_lengths)}, {np.max(episode_lengths)}]")
    
    print(f"\nReward Statistics:")
    print(f"  Mean: {np.mean(rewards_array):.4f}")
    print(f"  Std: {np.std(rewards_array):.4f}")
    print(f"  Min: {np.min(rewards_array):.4f}")
    print(f"  Max: {np.max(rewards_array):.4f}")
    print(f"  Non-zero: {np.count_nonzero(rewards_array) / len(rewards_array) * 100:.1f}%")
    print(f"  Positive: {np.sum(rewards_array > 0) / len(rewards_array) * 100:.1f}%")
    print(f"  Negative: {np.sum(rewards_array < 0) / len(rewards_array) * 100:.1f}%")
    
    print(f"\nAction Delta Statistics:")
    print(f"  Mean magnitude: {np.mean(np.linalg.norm(deltas_array, axis=1)):.6f}")
    print(f"  Std magnitude: {np.std(np.linalg.norm(deltas_array, axis=1)):.6f}")
    print(f"  Max magnitude: {np.max(np.linalg.norm(deltas_array, axis=1)):.6f}")
    print(f"  Non-zero: {np.count_nonzero(deltas_array) / deltas_array.size * 100:.1f}%")
    print(f"  Mean per dimension: {np.mean(np.abs(deltas_array), axis=0)}")
    
    print(f"\nFinal Action Statistics:")
    print(f"  Mean magnitude: {np.mean(np.linalg.norm(actions_array, axis=1)):.4f}")
    print(f"  Clipped: {np.sum((actions_array == -1.0) | (actions_array == 1.0)) / actions_array.size * 100:.1f}%")
    
    return {
        "rewards": rewards_array,
        "deltas": deltas_array,
        "actions": actions_array,
        "episode_lengths": episode_lengths
    }


def main():
    """Run diagnostic tests."""
    # Determine input/output sizes
    env = EdonHumanoidEnv(seed=42, profile="high_stress")
    test_obs = env.reset()
    test_baseline = baseline_controller(test_obs, edon_state=None)
    test_input = pack_observation(test_obs, np.array(test_baseline))
    input_size = len(test_input)
    output_size = len(test_baseline)
    
    # Test 1: Random policy
    print("\n" + "="*70)
    print("TEST 1: Random Initialized Policy")
    print("="*70)
    random_policy = EdonV7Policy(input_size, output_size)
    random_policy.eval()
    random_stats = test_policy(random_policy, "Random Policy", num_episodes=5)
    
    # Test 2: Trained policy
    print("\n" + "="*70)
    print("TEST 2: Trained v7 Policy")
    print("="*70)
    trained_policy = EdonV7Policy(input_size, output_size)
    model_path = Path("models/edon_v7_target48_v3.pt")
    
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        trained_policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"Loaded model from {model_path}")
    else:
        print(f"WARNING: Model not found at {model_path}, using random weights")
    
    trained_policy.eval()
    trained_stats = test_policy(trained_policy, "Trained Policy", num_episodes=5)
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON: Random vs Trained")
    print("="*70)
    
    random_delta_mag = np.mean(np.linalg.norm(random_stats["deltas"], axis=1))
    trained_delta_mag = np.mean(np.linalg.norm(trained_stats["deltas"], axis=1))
    
    random_reward_mean = np.mean(random_stats["rewards"])
    trained_reward_mean = np.mean(trained_stats["rewards"])
    
    random_length = np.mean(random_stats["episode_lengths"])
    trained_length = np.mean(trained_stats["episode_lengths"])
    
    print(f"\nDelta Magnitude:")
    print(f"  Random: {random_delta_mag:.6f}")
    print(f"  Trained: {trained_delta_mag:.6f}")
    print(f"  Difference: {trained_delta_mag - random_delta_mag:.6f} ({((trained_delta_mag - random_delta_mag) / (random_delta_mag + 1e-8) * 100):.1f}%)")
    
    print(f"\nReward Mean:")
    print(f"  Random: {random_reward_mean:.4f}")
    print(f"  Trained: {trained_reward_mean:.4f}")
    print(f"  Difference: {trained_reward_mean - random_reward_mean:.4f}")
    
    print(f"\nEpisode Length:")
    print(f"  Random: {random_length:.1f}")
    print(f"  Trained: {trained_length:.1f}")
    print(f"  Difference: {trained_length - random_length:.1f}")
    
    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    delta_change = abs(trained_delta_mag - random_delta_mag) / (random_delta_mag + 1e-8)
    reward_change = abs(trained_reward_mean - random_reward_mean)
    length_change = abs(trained_length - random_length)
    
    if delta_change < 0.01 and reward_change < 0.1 and length_change < 10:
        print("❌ TRAINING IS A NO-OP")
        print("   Policy behavior is essentially unchanged from random initialization.")
        print("   Possible causes:")
        print("   - Rewards too sparse/weak")
        print("   - Learning rate too low")
        print("   - Policy updates not effective")
        print("   - Observation space insufficient")
    elif delta_change < 0.1:
        print("⚠️  TRAINING HAS MINIMAL EFFECT")
        print("   Policy changed slightly but not meaningfully.")
    else:
        print("✅ TRAINING IS WORKING")
        print("   Policy behavior changed significantly from random initialization.")


if __name__ == "__main__":
    main()

