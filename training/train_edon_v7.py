"""
EDON v7 PPO Training

Trains an outcome-optimized EDON controller using Proximal Policy Optimization (PPO).

The v7 policy learns to maximize EDON episode score by optimizing per-step rewards
from training.edon_score.step_reward.

Architecture:
- Policy network: MLP that outputs action deltas (same as v6)
- Value network: MLP that estimates state value
- PPO algorithm with clipping and value loss
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.edon_humanoid_env import EdonHumanoidEnv
from training.edon_score import compute_episode_score, step_reward
from run_eval import baseline_controller


class EdonV7Policy(nn.Module):
    """
    EDON v7 Policy Network (MLP with tanh-squashed actions).
    
    Uses tanh squashing to bound actions smoothly (no hard clipping).
    """
    
    def __init__(self, input_size: int, output_size: int, max_delta: float = 1.0, hidden_sizes: List[int] = [128, 128, 64]):
        super().__init__()
        
        # Shared network
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.net = nn.Sequential(*layers)
        
        # Action head (mean)
        self.mu_head = nn.Linear(prev_size, output_size)
        
        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.zeros(output_size))
        
        self.max_delta = max_delta
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: output mean and std for action distribution.
        
        Returns:
            (mu, std) - mean and standard deviation
        """
        x = self.net(x)
        mu = self.mu_head(x)
        std = self.log_std.exp().clamp(min=1e-6, max=1.0)  # Clamp std for stability
        return mu, std
    
    def sample_action(self, obs: torch.Tensor, use_rsample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy with tanh squashing.
        
        Args:
            obs: Observations
            use_rsample: If True, use rsample() (for gradients). If False, use sample() (no gradients).
        
        Returns:
            (action, log_prob) - bounded action and its log probability
        """
        mu, std = self.forward(obs)
        dist = Normal(mu, std)
        
        # Sample from normal distribution
        if use_rsample:
            raw_action = dist.rsample()  # Reparameterized sample (for gradients)
        else:
            raw_action = dist.sample()  # Regular sample (no gradients)
        
        log_prob_raw = dist.log_prob(raw_action).sum(dim=-1)
        
        # Tanh squashing (smooth bounded action, no hard clip)
        action = torch.tanh(raw_action) * self.max_delta
        
        # Tanh correction: log_prob = log_prob_raw - log(1 - tanh^2(x))
        # This accounts for the change of variables from raw_action to action
        action_normalized = action / self.max_delta
        tanh_correction = -torch.log(1 - action_normalized.pow(2) + 1e-6).sum(dim=-1)
        log_prob = log_prob_raw + tanh_correction
        
        return action, log_prob
    
    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of action under current policy.
        
        Args:
            obs: Observations
            action: Actions (already tanh-squashed, in range [-max_delta, max_delta])
            
        Returns:
            Log probabilities
        """
        mu, std = self.forward(obs)
        dist = Normal(mu, std)
        
        # Inverse tanh to get raw action (before tanh squashing)
        # action is in [-max_delta, max_delta], need to map to [-1, 1] first
        action_normalized = action / self.max_delta
        raw_action = torch.atanh(torch.clamp(action_normalized, -0.999, 0.999))
        
        # Log prob of raw action under normal distribution
        log_prob_raw = dist.log_prob(raw_action).sum(dim=-1)
        
        # Tanh correction: log_prob = log_prob_raw - log(1 - tanh^2(x))
        # d/dx tanh(x) = 1 - tanh^2(x), so correction is -log(1 - tanh^2(x))
        tanh_correction = -torch.log(1 - action_normalized.pow(2) + 1e-6).sum(dim=-1)
        log_prob = log_prob_raw + tanh_correction
        
        return log_prob


class EdonV7Value(nn.Module):
    """
    EDON v7 Value Network (MLP).
    
    Estimates state value for PPO.
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 128, 64]):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer (single value)
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: output state value."""
        return self.network(x).squeeze(-1)


def pack_observation(obs: Dict[str, Any], baseline_action: np.ndarray) -> np.ndarray:
    """
    Pack observation into input vector for policy network.
    
    Uses similar structure to v6 but simplified for RL.
    """
    # Extract key features
    roll = float(obs.get("roll", 0.0))
    pitch = float(obs.get("pitch", 0.0))
    roll_velocity = float(obs.get("roll_velocity", 0.0))
    pitch_velocity = float(obs.get("pitch_velocity", 0.0))
    com_x = float(obs.get("com_x", 0.0))
    com_y = float(obs.get("com_y", 0.0))
    com_velocity_x = float(obs.get("com_velocity_x", 0.0))
    com_velocity_y = float(obs.get("com_velocity_y", 0.0))
    
    # Compute derived features
    tilt_mag = np.sqrt(roll**2 + pitch**2)
    vel_norm = np.sqrt(roll_velocity**2 + pitch_velocity**2)
    com_norm = np.sqrt(com_x**2 + com_y**2)
    com_vel_norm = np.sqrt(com_velocity_x**2 + com_velocity_y**2)
    
    # Pack: features + baseline action
    input_vec = np.concatenate([
        [roll, pitch, roll_velocity, pitch_velocity],
        [com_x, com_y, com_velocity_x, com_velocity_y],
        [tilt_mag, vel_norm, com_norm, com_vel_norm],
        baseline_action.astype(np.float32)
    ])
    
    return input_vec


class PPO:
    """
    Proximal Policy Optimization (PPO) algorithm for EDON v7.
    """
    
    def __init__(
        self,
        policy: EdonV7Policy,
        value: EdonV7Value,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5
    ):
        self.policy = policy
        self.value = value
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.policy_optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(value.parameters(), lr=lr)
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float = 0.0
    ) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Returns:
            (advantages, returns)
        """
        advantages = []
        returns = []
        
        gae = 0.0
        next_value = next_value
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * next_value - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            next_value = values[t]
        
        return advantages, returns
    
    def update(
        self,
        obs_batch: List[np.ndarray],
        action_batch: List[np.ndarray],
        old_log_probs: List[float],
        advantages: List[float],
        returns: List[float],
        epochs: int = 4
    ) -> Dict[str, float]:
        """
        Update policy and value networks using PPO.
        
        Args:
            obs_batch: Batch of observations
            action_batch: Batch of actions taken
            old_log_probs: Log probabilities of actions under old policy
            advantages: Computed advantages
            returns: Computed returns
            epochs: Number of update epochs
            
        Returns:
            Dict with diagnostic metrics (losses, KL, entropy, etc.)
        """
        device = next(self.policy.parameters()).device
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(obs_batch)).to(device)
        action_tensor = torch.FloatTensor(np.array(action_batch)).to(device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(device)
        advantages_tensor = torch.FloatTensor(advantages).to(device)
        returns_tensor = torch.FloatTensor(returns).to(device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Track metrics across epochs
        epoch_metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl_div": [],
            "clip_fraction": []
        }
        
        for epoch in range(epochs):
            # Forward pass - get new log probs from current policy
            new_log_probs = self.policy.log_prob(obs_tensor, action_tensor)
            values = self.value(obs_tensor)
            
            # Compute KL divergence (old vs new policy)
            with torch.no_grad():
                kl_div = (old_log_probs_tensor - new_log_probs).mean()
                # Clamp to reasonable range for logging
                kl_div_clamped = torch.clamp(kl_div, -10.0, 10.0).item()
            
            # Compute policy loss (PPO clipped objective)
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            policy_loss_1 = ratio * advantages_tensor
            policy_loss_2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
            
            # Clip fraction (how often we're clipping)
            clip_fraction = ((ratio < (1 - self.clip_epsilon)) | (ratio > (1 + self.clip_epsilon))).float().mean()
            
            # Value loss
            value_loss = nn.functional.mse_loss(values, returns_tensor)
            
            # Entropy bonus (encourage exploration) - compute from policy
            mu, std = self.policy.forward(obs_tensor)
            dist = Normal(mu, std)
            entropy = dist.entropy().sum(dim=-1).mean()
            
            # Track metrics
            epoch_metrics["policy_loss"].append(policy_loss.item())
            epoch_metrics["value_loss"].append(value_loss.item())
            epoch_metrics["entropy"].append(entropy.item())
            epoch_metrics["kl_div"].append(kl_div_clamped)
            epoch_metrics["clip_fraction"].append(clip_fraction.item())
            
            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()
            
            # Update value
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
            self.value_optimizer.step()
        
        # Return average metrics across epochs
        return {
            "policy_loss": np.mean(epoch_metrics["policy_loss"]),
            "value_loss": np.mean(epoch_metrics["value_loss"]),
            "entropy": np.mean(epoch_metrics["entropy"]),
            "kl_div": np.mean(epoch_metrics["kl_div"]),
            "clip_fraction": np.mean(epoch_metrics["clip_fraction"]),
            "policy_loss_std": np.std(epoch_metrics["policy_loss"]),
            "value_loss_std": np.std(epoch_metrics["value_loss"])
        }


def collect_trajectory(
    env: EdonHumanoidEnv,
    policy: EdonV7Policy,
    value_net: EdonV7Value,
    baseline_controller_fn,
    max_steps: int = 1000,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Collect a single trajectory using current policy.
    
    Returns:
        Trajectory dict with observations, actions, rewards, etc.
    """
    try:
        obs = env.reset()
    except Exception as e:
        print(f"[COLLECT] Error in env.reset(): {e}")
        import traceback
        traceback.print_exc()
        raise
    
    trajectory = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "values": [],
        "log_probs": []
    }
    
    # Use eval mode for consistent behavior during collection
    policy.eval()
    
    step = 0
    done = False
    
    with torch.no_grad():  # No gradients needed during collection
        while not done and step < max_steps:
            try:
                # Get baseline action
                baseline_action = baseline_controller_fn(obs, edon_state=None)
                baseline_action = np.array(baseline_action)
                
                # Pack observation
                obs_vec = pack_observation(obs, baseline_action)
                obs_tensor = torch.FloatTensor(obs_vec).unsqueeze(0).to(device)
                
                # Sample action from policy (tanh-squashed, bounded)
                # Use sample() instead of rsample() during collection (no gradients needed)
                action_delta_tensor, log_prob_tensor = policy.sample_action(obs_tensor, use_rsample=False)
                action_delta = action_delta_tensor.cpu().numpy()[0]
                log_prob = log_prob_tensor.item()
                
                # Get value estimate
                value_tensor = value_net(obs_tensor)
                value_est = value_tensor.item()
                
                # Compute final action (baseline + delta, no need to clip - already bounded by tanh)
                final_action = baseline_action + action_delta
                # Still clip to [-1, 1] for safety (but should rarely be needed with tanh)
                final_action = np.clip(final_action, -1.0, 1.0)
                
                # Store baseline_action in env for delta computation
                if hasattr(env, 'last_baseline_action'):
                    env.last_baseline_action = baseline_action
                
                # Step environment
                next_obs, reward, done, info = env.step(final_action)
                
                # Store trajectory
                trajectory["observations"].append(obs_vec)
                trajectory["actions"].append(action_delta)
                trajectory["rewards"].append(reward)
                trajectory["dones"].append(done)
                trajectory["values"].append(value_est)
                trajectory["log_probs"].append(log_prob)
                
                obs = next_obs
                step += 1
                
            except Exception as e:
                print(f"[COLLECT] Error at step {step}: {e}")
                import traceback
                traceback.print_exc()
                break
    
    return trajectory


def main():
    """Main PPO training loop."""
    parser = argparse.ArgumentParser(description="EDON v7 PPO Training")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=1000,
                       help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--profile", type=str, default="high_stress",
                       choices=["light_stress", "medium_stress", "high_stress", "hell_stress"],
                       help="Stress profile")
    parser.add_argument("--lr", type=float, default=5e-4,
                       help="Learning rate (increased for bolder updates)")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for updates")
    parser.add_argument("--update-epochs", type=int, default=12,
                       help="Number of PPO update epochs (increased for bolder updates)")
    parser.add_argument("--output-dir", type=str, default="models",
                       help="Output directory for model")
    parser.add_argument("--model-name", type=str, default="edon_v7",
                       help="Model filename (without extension)")
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create environment
    env = EdonHumanoidEnv(seed=args.seed, profile=args.profile)
    
    # Determine input/output sizes
    test_obs = env.reset()
    test_baseline = baseline_controller(test_obs, edon_state=None)
    test_input = pack_observation(test_obs, np.array(test_baseline))
    input_size = len(test_input)
    output_size = len(test_baseline)
    
    print(f"[EDON-V7] Input size: {input_size}, Output size: {output_size}")
    
    # Create networks
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[EDON-V7] Using device: {device}")
    
    policy = EdonV7Policy(input_size, output_size).to(device)
    value = EdonV7Value(input_size).to(device)
    
    # Value network is separate
    
    # Create PPO
    ppo = PPO(policy, value, lr=args.lr, gamma=args.gamma)
    
    # Training loop
    print(f"\n[EDON-V7] Starting training: {args.episodes} episodes, profile={args.profile}")
    print("="*70)
    
    episode_scores = []
    episode_rewards = []
    episode_lengths = []
    episode_interventions = []
    rolling_rewards = deque(maxlen=20)  # For rolling mean
    
    for episode in range(args.episodes):
        try:
            # Collect trajectory
            trajectory = collect_trajectory(
                env=env,
                policy=policy,
                value_net=value,
                baseline_controller_fn=baseline_controller,
                max_steps=args.max_steps,
                device=device
            )
        except Exception as e:
            print(f"[TRAIN] Error collecting trajectory: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        print(f"[TRAIN] Computing GAE for {len(trajectory['rewards'])} steps")
        try:
            # CRITICAL FIX: Normalize rewards to fix value function scale mismatch
            # Rewards are in [-50, 2] but value function initialized near 0
            # This causes value loss to be 70k+ instead of < 100
            rewards_array = np.array(trajectory["rewards"])
            if len(rewards_array) > 0:
                reward_mean = float(np.mean(rewards_array))
                reward_std = float(np.std(rewards_array))
                if reward_std < 1e-6:
                    reward_std = 1.0
                # Normalize rewards (z-score)
                normalized_rewards = [(r - reward_mean) / reward_std for r in trajectory["rewards"]]
                # Normalize values too
                normalized_values = [(v - reward_mean) / reward_std for v in trajectory["values"]]
                # Normalize next_value
                next_value_raw = value(torch.FloatTensor(trajectory["observations"][-1]).unsqueeze(0).to(device)).item() if len(trajectory["observations"]) > 0 else 0.0
                normalized_next_value = (next_value_raw - reward_mean) / reward_std
            else:
                normalized_rewards = trajectory["rewards"]
                normalized_values = trajectory["values"]
                normalized_next_value = 0.0
            
            # Compute returns and advantages with normalized rewards/values
            advantages, returns = ppo.compute_gae(
                rewards=normalized_rewards,
                values=normalized_values,
                dones=trajectory["dones"],
                next_value=normalized_next_value
            )
        except Exception as e:
            print(f"[TRAIN] Error computing GAE: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Update policy and get diagnostic metrics
        update_metrics = {}
        if len(trajectory["observations"]) > 0:
            try:
                update_metrics = ppo.update(
                    obs_batch=trajectory["observations"],
                    action_batch=trajectory["actions"],
                    old_log_probs=trajectory["log_probs"],
                    advantages=advantages,
                    returns=returns,
                    epochs=args.update_epochs
                )
            except Exception as e:
                print(f"[TRAIN] Error updating policy: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Compute episode metrics
        total_reward = sum(trajectory["rewards"])
        episode_length = len(trajectory["rewards"])
        interventions = sum(1 for d in trajectory["dones"] if d)
        
        # Reward statistics
        rewards_array = np.array(trajectory["rewards"])
        reward_mean = float(np.mean(rewards_array))
        reward_std = float(np.std(rewards_array))
        reward_min = float(np.min(rewards_array))
        reward_max = float(np.max(rewards_array))
        reward_nonzero = float(np.count_nonzero(rewards_array) / len(rewards_array) * 100)
        
        # Compute average instability (if available in trajectory info)
        # For now, estimate from reward variance (higher variance = more instability)
        avg_instability = float(reward_std / 10.0) if reward_std > 0 else 0.0  # Rough proxy
        
        # Compute EDON episode score
        episode_summary = {
            "interventions": interventions,
            "stability_score": 0.0,  # Would need to track this properly
            "episode_length": episode_length
        }
        episode_score = compute_episode_score(episode_summary)
        
        episode_scores.append(episode_score)
        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        episode_interventions.append(interventions)
        rolling_rewards.append(total_reward)
        rolling_mean_reward = float(np.mean(list(rolling_rewards))) if len(rolling_rewards) > 0 else total_reward
        
        # Print progress with diagnostics - EVERY EPISODE
        print(
            f"[V7] ep={episode + 1}/{args.episodes} "
            f"reward={total_reward:.2f} "
            f"len={episode_length} "
            f"interventions={interventions} "
            f"avg_instab={avg_instability:.2f} "
            f"rolling_mean={rolling_mean_reward:.2f}"
        )
        
        # Detailed PPO metrics (every episode)
        if update_metrics:
            print(
                f"  PPO: policy_loss={update_metrics['policy_loss']:.4e}, "
                f"value_loss={update_metrics['value_loss']:.4e}, "
                f"entropy={update_metrics['entropy']:.4e}, "
                f"kl={update_metrics['kl_div']:.4e}, "
                f"clip_frac={update_metrics['clip_fraction']:.2%}"
            )
        
        # Summary every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_score_last_10 = np.mean(episode_scores[-10:]) if len(episode_scores) >= 10 else episode_score
            avg_reward_last_10 = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else total_reward
            avg_length_last_10 = np.mean(episode_lengths[-10:]) if len(episode_lengths) >= 10 else episode_length
            avg_interventions_last_10 = np.mean(episode_interventions[-10:]) if len(episode_interventions) >= 10 else interventions
            
            print(
                f"\n[V7-SUMMARY] Episodes {episode + 1 - 9}-{episode + 1}: "
                f"avg_score={avg_score_last_10:.2f}, "
                f"avg_reward={avg_reward_last_10:.2f}, "
                f"avg_length={avg_length_last_10:.1f}, "
                f"avg_interventions={avg_interventions_last_10:.1f}"
            )
            print(
                f"  Rewards: mean={reward_mean:.3f}, std={reward_std:.3f}, "
                f"range=[{reward_min:.3f}, {reward_max:.3f}], "
                f"nonzero={reward_nonzero:.1f}%"
            )
            print()  # Blank line for readability
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{args.model_name}.pt"
    
    print(f"\n[EDON-V7] Saving model to {model_path}")
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "value_state_dict": value.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "max_delta": policy.max_delta,
        "episodes": args.episodes,
        "final_avg_score": np.mean(episode_scores[-10:]) if len(episode_scores) >= 10 else np.mean(episode_scores)
    }, model_path)
    
    print(f"[EDON-V7] Training complete!")
    print(f"  Final average score (last 10): {np.mean(episode_scores[-10:]):.2f}")
    print(f"  Final average reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
    print(f"  Model saved to: {model_path}")


if __name__ == "__main__":
    main()

