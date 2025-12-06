"""
Debug v8 Policy Learning Issues

Analyzes why policy loss = 0 and policy isn't learning.
Checks: advantages, rewards, gradients, etc.
"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.edon_humanoid_env_v8 import EdonHumanoidEnvV8
from training.edon_v8_policy import EdonV8StrategyPolicy, pack_observation_v8
from training.fail_risk_model import FailRiskModel
from run_eval import baseline_controller, make_humanoid_env
from env.edon_humanoid_env import EdonHumanoidEnv


def debug_policy_learning():
    """Debug why v8 policy isn't learning."""
    print("="*80)
    print("Debugging v8 Policy Learning")
    print("="*80)
    
    # Create environment
    base_env = make_humanoid_env(seed=0, profile="high_stress")
    env_wrapper = EdonHumanoidEnv(base_env=base_env, seed=0, profile="high_stress")
    
    # Load fail-risk model
    fail_risk_model = None
    fail_risk_model_path = Path("models/edon_fail_risk_v1_fixed_v2.pt")
    if fail_risk_model_path.exists():
        checkpoint = torch.load(fail_risk_model_path, map_location="cpu", weights_only=False)
        input_size = checkpoint.get("input_size", 15)
        fail_risk_model = FailRiskModel(input_size=input_size)
        fail_risk_model.load_state_dict(checkpoint["model_state_dict"])
        fail_risk_model.eval()
    
    # Load strategy policy
    strategy_model_path = Path("models/edon_v8_strategy_v1_no_reflex.pt")
    if not strategy_model_path.exists():
        print(f"[ERROR] Strategy model not found: {strategy_model_path}")
        return
    
    checkpoint = torch.load(strategy_model_path, map_location="cpu", weights_only=False)
    input_size = checkpoint.get("input_size", 25)
    policy = EdonV8StrategyPolicy(input_size=input_size)
    
    # Try different checkpoint formats
    if "policy_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["policy_state_dict"])
    elif "model_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["model_state_dict"])
    else:
        policy.load_state_dict(checkpoint)
    
    policy.eval()
    
    # Create v8 environment
    env = EdonHumanoidEnvV8(
        strategy_policy=policy,
        fail_risk_model=fail_risk_model,
        base_env=env_wrapper,
        seed=0,
        profile="high_stress"
    )
    
    # Collect a short trajectory
    print("\n[1] Collecting trajectory...")
    obs = env.reset()
    rewards = []
    observations = []
    strategies = []
    log_probs = []
    
    for step in range(100):
        # Get baseline
        baseline_action = baseline_controller(obs, edon_state=None)
        baseline_action = np.array(baseline_action)
        
        # Pack observation
        obs_vec = pack_observation_v8(
            obs=obs,
            baseline_action=baseline_action,
            fail_risk=env.current_fail_risk,
            instability_score=0.0,
            phase="stable"
        )
        
        # Get policy output
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_vec).unsqueeze(0)
            strategy_id, modulations, log_prob = policy.sample_action(obs_tensor, deterministic=False)
        
        # Step environment
        next_obs, reward, done, info = env.step(edon_core_state=None)
        
        rewards.append(reward)
        observations.append(obs_vec)
        strategies.append(strategy_id)
        log_probs.append(log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob)
        
        obs = next_obs
        if done:
            break
    
    print(f"  Collected {len(rewards)} steps")
    print(f"  Total reward: {sum(rewards):.2f}")
    print(f"  Mean reward: {np.mean(rewards):.2f}")
    print(f"  Std reward: {np.std(rewards):.2f}")
    print(f"  Min reward: {np.min(rewards):.2f}")
    print(f"  Max reward: {np.max(rewards):.2f}")
    
    # Compute returns
    print("\n[2] Computing returns...")
    from training.train_edon_v8_strategy import PPO
    ppo = PPO(policy, lr=5e-4, gamma=0.995)
    
    dones = [False] * len(rewards)
    returns = ppo.compute_returns(rewards, dones)
    
    print(f"  Mean return: {np.mean(returns):.2f}")
    print(f"  Std return: {np.std(returns):.2f}")
    print(f"  Min return: {np.min(returns):.2f}")
    print(f"  Max return: {np.max(returns):.2f}")
    
    # Compute advantages (GAE)
    print("\n[3] Computing advantages (GAE)...")
    advantages, returns_gae = ppo.compute_gae(
        rewards=rewards,
        values=returns,  # Use returns as value estimates
        dones=dones,
        next_value=0.0
    )
    
    print(f"  Mean advantage: {np.mean(advantages):.2f}")
    print(f"  Std advantage: {np.std(advantages):.2f}")
    print(f"  Min advantage: {np.min(advantages):.2f}")
    print(f"  Max advantage: {np.max(advantages):.2f}")
    
    # Check if advantages are too small
    if np.abs(np.mean(advantages)) < 0.01:
        print("  [WARNING] Mean advantage is very small - policy won't learn!")
    
    if np.std(advantages) < 0.01:
        print("  [WARNING] Advantage std is very small - no signal to learn from!")
    
    # Check log probs
    print("\n[4] Analyzing log probabilities...")
    print(f"  Mean log_prob: {np.mean(log_probs):.4f}")
    print(f"  Std log_prob: {np.std(log_probs):.4f}")
    print(f"  Min log_prob: {np.min(log_probs):.4f}")
    print(f"  Max log_prob: {np.max(log_probs):.4f}")
    
    # Check strategy distribution
    print("\n[5] Strategy distribution...")
    from collections import Counter
    strategy_counts = Counter(strategies)
    for strategy_id, count in strategy_counts.most_common():
        strategy_name = ["NORMAL", "HIGH_DAMPING", "RECOVERY_BALANCE", "COMPLIANT_TERRAIN"][strategy_id]
        print(f"  {strategy_name}: {count} ({100*count/len(strategies):.1f}%)")
    
    # Try a policy update to see what happens
    print("\n[6] Testing policy update...")
    policy.train()
    
    obs_tensor = torch.FloatTensor(np.array(observations))
    old_log_probs_tensor = torch.FloatTensor(log_probs)
    advantages_tensor = torch.FloatTensor(advantages)
    returns_tensor = torch.FloatTensor(returns_gae)
    
    # Normalize advantages
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
    
    # Get current policy outputs
    strategy_logits, _ = policy(obs_tensor)
    strategy_dist = torch.distributions.Categorical(logits=strategy_logits)
    strategy_tensor = torch.LongTensor(strategies)
    new_log_probs = strategy_dist.log_prob(strategy_tensor)
    
    # Compute ratio
    ratio = torch.exp(new_log_probs - old_log_probs_tensor)
    
    # PPO clip
    clipped_ratio = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
    policy_loss = -torch.min(ratio * advantages_tensor, clipped_ratio * advantages_tensor).mean()
    
    print(f"  Policy loss: {policy_loss.item():.6f}")
    print(f"  Ratio mean: {ratio.mean().item():.4f}")
    print(f"  Ratio std: {ratio.std().item():.4f}")
    print(f"  Ratio min: {ratio.min().item():.4f}")
    print(f"  Ratio max: {ratio.max().item():.4f}")
    print(f"  Advantages mean (normalized): {advantages_tensor.mean().item():.4f}")
    print(f"  Advantages std (normalized): {advantages_tensor.std().item():.4f}")
    
    if abs(policy_loss.item()) < 1e-6:
        print("  [WARNING] Policy loss is essentially zero!")
        print("  Possible causes:")
        print("    - Advantages are too small")
        print("    - Policy hasn't changed (ratio ~ 1.0)")
        print("    - Normalized advantages are zero")
    
    # Check gradients
    print("\n[7] Checking gradients...")
    policy_loss.backward()
    
    total_grad_norm = 0.0
    param_count = 0
    for name, param in policy.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm
            param_count += 1
            if grad_norm > 0.001:  # Only print significant gradients
                print(f"  {name}: grad_norm = {grad_norm:.6f}")
    
    if param_count > 0:
        avg_grad_norm = total_grad_norm / param_count
        print(f"  Average grad norm: {avg_grad_norm:.6f}")
        if avg_grad_norm < 1e-6:
            print("  [WARNING] Gradients are very small - policy won't update!")
    else:
        print("  [ERROR] No gradients computed!")
    
    print("\n" + "="*80)
    print("Debugging Complete")
    print("="*80)


if __name__ == "__main__":
    debug_policy_learning()

