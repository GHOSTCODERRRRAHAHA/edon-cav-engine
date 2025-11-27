"""
EDON v8 Strategy Policy Training

Trains the v8 strategy policy using PPO, similar to v7 but with:
- Strategy + modulation outputs (not raw deltas)
- v8 environment wrapper
- v8 metrics and scoring
- Intervention-first reward shaping

Reward Structure:
- Stability penalties scaled down ~10x (per-step ~-0.3 instead of ~-3.0)
- w_intervention default: 20.0 (increased from 5.0)
- Episode-level stability constraint remains intact as hard safety budget

Recommended Phase A (fast probe) command:
    python training/train_edon_v8_strategy.py \\
      --episodes 100 \\
      --profile high_stress \\
      --seed 0 \\
      --lr 5e-4 \\
      --gamma 0.995 \\
      --update-epochs 10 \\
      --output-dir models \\
      --model-name edon_v8_strategy_intervention_first_w20_phaseA \\
      --fail-risk-model models/edon_fail_risk_v1_fixed_v2.pt \\
      --max-steps 1000 \\
      --w-intervention 20.0 \\
      --w-stability 1.0 \\
      --w-torque 0.1
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.edon_humanoid_env_v8 import EdonHumanoidEnvV8
from training.edon_v8_policy import EdonV8StrategyPolicy, pack_observation_v8
from training.fail_risk_model import FailRiskModel
from training.edon_score import compute_episode_score
from metrics.edon_v8_metrics import compute_episode_score_v8, compute_episode_metrics_v8
from run_eval import baseline_controller


class PPO:
    """PPO algorithm for v8 strategy policy."""
    
    def __init__(
        self,
        policy: EdonV8StrategyPolicy,
        lr: float = 5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.02  # Increased from 0.01 for more exploration
    ):
        self.policy = policy
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    def compute_returns(self, rewards: List[float], dones: List[bool], next_value: float = 0.0) -> List[float]:
        """Compute discounted returns."""
        returns = []
        G = next_value
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0.0
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns
    
    def compute_gae(
        self,
        rewards: List[float],
        values: Optional[List[float]] = None,
        dones: Optional[List[bool]] = None,
        next_value: float = 0.0
    ) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        If values not provided, uses returns as value estimates (simpler baseline).
        """
        if values is None or len(values) != len(rewards):
            # Compute returns and use them as value estimates
            returns_list = self.compute_returns(rewards, dones or [False] * len(rewards), next_value)
            values = returns_list
        
        if dones is None:
            dones = [False] * len(rewards)
        
        advantages = []
        returns = []
        
        gae = 0.0
        next_value_est = next_value
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
                next_value_est = 0.0
            else:
                next_val = values[t + 1] if t + 1 < len(values) else next_value_est
                delta = rewards[t] + self.gamma * next_val - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
                next_value_est = values[t]
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def update(
        self,
        obs_batch: List[np.ndarray],
        strategy_batch: List[int],
        modulations_batch: List[Dict[str, float]],
        old_log_probs: List[float],
        returns: List[float],
        advantages: List[float],
        epochs: int = 10
    ) -> Dict[str, float]:
        """Update policy with PPO."""
        obs_tensor = torch.FloatTensor(np.array(obs_batch))
        returns_tensor = torch.FloatTensor(returns)
        advantages_tensor = torch.FloatTensor(advantages)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        total_policy_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        
        for _ in range(epochs):
            self.optimizer.zero_grad()
            
            # Get current policy outputs
            strategy_logits, _ = self.policy(obs_tensor)
            strategy_dist = Categorical(logits=strategy_logits)
            
            # Compute log probs for sampled strategies
            strategy_tensor = torch.LongTensor(strategy_batch)
            log_probs = strategy_dist.log_prob(strategy_tensor)
            
            # Compute entropy
            entropy = strategy_dist.entropy().mean()
            
            # Compute KL divergence (approximate - would need old policy logits for exact KL)
            # For now, use a simple approximation based on log prob differences
            kl = (log_probs - old_log_probs_tensor).mean()
            
            # PPO clip
            ratio = torch.exp(log_probs - old_log_probs_tensor)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            policy_loss = -torch.min(ratio * advantages_tensor, clipped_ratio * advantages_tensor).mean()
            
            # Add entropy bonus for exploration
            total_loss = policy_loss - self.entropy_coef * entropy
            
            total_loss.backward()
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_entropy += entropy.item()
            total_kl += kl.item()
        
        return {
            "policy_loss": total_policy_loss / epochs,
            "entropy": total_entropy / epochs,
            "kl_div": total_kl / epochs
        }


def collect_trajectory(
    env: EdonHumanoidEnvV8,
    policy: EdonV8StrategyPolicy,
    max_steps: int = 1000,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Collect trajectory using v8 policy."""
    obs = env.reset()
    observations = []  # Observation vectors for policy
    observations_dict = []  # Actual observation dicts for reward computation
    strategies = []
    modulations_list = []
    log_probs = []
    rewards = []
    dones = []
    infos = []
    
    step = 0
    while step < max_steps:
        # Get baseline for packing
        baseline_action = baseline_controller(obs, edon_state=None)
        baseline_action = np.array(baseline_action)
        
        # Pack observation with stacking (env handles this internally now)
        # The env's step() method will use pack_stacked_observation_v8
        # But we need to get the packed vector for the trajectory
        # Actually, the env already packs it in step(), so we need to get it from there
        # For now, let's pack it here too for trajectory collection
        from training.edon_v8_policy import pack_stacked_observation_v8
        obs_vec = pack_stacked_observation_v8(
            obs=obs,
            baseline_action=baseline_action,
            fail_risk=env.current_fail_risk,
            instability_score=0.0,
            phase="stable",
            obs_history=list(env.obs_history),
            near_fail_history=list(env.near_fail_history),
            obs_vec_history=list(env.obs_vec_history),
            stack_size=8
        )
        
        # Sample from policy
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_vec).unsqueeze(0).to(device)
            strategy_id, modulations, log_prob = policy.sample_action(obs_tensor, deterministic=False)
        
        # Step environment (v8 env handles strategy + reflex internally)
        next_obs, reward, done, info = env.step(edon_core_state=None)
        
        # Print progress every 100 steps for long episodes
        if step % 100 == 0 and step > 0:
            print(f"[COLLECT] Step {step}/{max_steps}, reward={reward:.3f}, done={done}", flush=True)
        
        # Store trajectory
        observations.append(obs_vec)  # For policy input
        observations_dict.append(obs)  # Actual obs dict for reward computation
        strategies.append(strategy_id)
        modulations_list.append(modulations)
        log_probs.append(log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob)
        rewards.append(reward)
        dones.append(done)
        infos.append(info)
        
        obs = next_obs
        step += 1
        
        if done:
            break
    
    return {
        "observations": observations,  # Observation vectors
        "observations_dict": observations_dict,  # Actual observation dicts
        "strategies": strategies,
        "modulations": modulations_list,
        "log_probs": log_probs,
        "rewards": rewards,
        "dones": dones,
        "infos": infos
    }


def main():
    """Main v8 training loop."""
    parser = argparse.ArgumentParser(description="EDON v8 Strategy Policy Training")
    parser.add_argument("--episodes", type=int, default=300, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--profile", type=str, default="high_stress", help="Stress profile")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    parser.add_argument("--update-epochs", type=int, default=10, help="PPO update epochs")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    parser.add_argument("--model-name", type=str, default="edon_v8_strategy_v1", help="Model filename")
    parser.add_argument("--fail-risk-model", type=str, default="models/edon_fail_risk_v1_fixed_v2.pt", help="Fail-risk model path")
    
    # Reward weights (intervention-first tuning)
    parser.add_argument("--w-intervention", type=float, default=20.0, help="Weight for intervention penalty (primary goal)")
    parser.add_argument("--w-stability", type=float, default=1.0, help="Weight for stability penalty (per-step, scaled down ~10x)")
    parser.add_argument("--w-torque", type=float, default=0.1, help="Weight for torque/action penalty")
    
    # Retroactive intervention penalty (credit assignment)
    parser.add_argument("--retroactive-steps", type=int, default=20, help="Number of steps before intervention to penalize retroactively")
    parser.add_argument("--w-retroactive", type=float, default=3.0, help="Weight for retroactive intervention penalty per step")
    
    # Stability constraint
    parser.add_argument("--stability-baseline", type=float, default=0.0206, help="Baseline stability for constraint (from baseline runs)")
    parser.add_argument("--stability-threshold-factor", type=float, default=1.05, help="Stability threshold factor (1.05 = 5% worse allowed)")
    parser.add_argument("--w-stability-episode", type=float, default=10.0, help="Weight for episode-level stability penalty")
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create environment
    from env.edon_humanoid_env import EdonHumanoidEnv
    base_env = EdonHumanoidEnv(seed=args.seed, profile=args.profile)
    
    # Load fail-risk model if available
    fail_risk_model = None
    try:
        if Path(args.fail_risk_model).exists():
            checkpoint = torch.load(args.fail_risk_model, map_location="cpu", weights_only=False)
            input_size = checkpoint.get("input_size", 15)
            fail_risk_model = FailRiskModel(input_size=input_size)
            fail_risk_model.load_state_dict(checkpoint["model_state_dict"])
            fail_risk_model.eval()
            print(f"[V8] Loaded fail-risk model from {args.fail_risk_model}")
    except Exception as e:
        print(f"[V8] Warning: Could not load fail-risk model: {e}")
    
    # Create a temporary v8 environment to infer obs_dim
    # We need to create a dummy policy first, but we'll recreate it with the correct size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[EDON-V8] Using device: {device}")
    
    # Create a temporary dummy policy just to create the env for obs_dim inference
    # We'll recreate the policy with the correct size after inferring obs_dim
    dummy_policy = EdonV8StrategyPolicy(input_size=248).to(device)  # Temporary dummy
    temp_env = EdonHumanoidEnvV8(
        strategy_policy=dummy_policy,
        fail_risk_model=fail_risk_model,
        base_env=base_env,
        seed=args.seed,
        profile=args.profile,
        device=device,
        w_intervention=args.w_intervention,
        w_stability=args.w_stability,
        w_torque=args.w_torque
    )
    
    # Infer obs_dim from environment
    obs_dim = EdonV8StrategyPolicy._infer_obs_dim_from_env(temp_env)
    print(f"[V8] Inferred obs_dim from env: {obs_dim}")
    
    # Create policy with inferred obs_dim (NOT loading any old checkpoints)
    policy = EdonV8StrategyPolicy(input_size=obs_dim).to(device)
    
    # Print obs_dim and first layer in_features
    first_layer = policy.feature_net[0]
    print(f"[V8] obs_dim: {obs_dim}")
    print(f"[V8] First Linear layer in_features: {first_layer.in_features}")
    
    # Verify the policy was created correctly
    if first_layer.in_features != obs_dim:
        raise ValueError(f"Policy first layer in_features ({first_layer.in_features}) != inferred obs_dim ({obs_dim})!")
    
    # Recreate v8 environment with the correct policy
    env = EdonHumanoidEnvV8(
        strategy_policy=policy,
        fail_risk_model=fail_risk_model,
        base_env=base_env,
        seed=args.seed,
        profile=args.profile,
        device=device,
        w_intervention=args.w_intervention,
        w_stability=args.w_stability,
        w_torque=args.w_torque
    )
    
    # Create PPO
    ppo = PPO(policy, lr=args.lr, gamma=args.gamma)
    
    # Print reward configuration
    print(f"\n[EDON-V8] Reward weights:")
    print(f"  W_INTERVENTION = {args.w_intervention} (primary goal)")
    print(f"  W_STABILITY = {args.w_stability}")
    print(f"  W_TORQUE = {args.w_torque}")
    print(f"  Stability baseline = {args.stability_baseline}")
    print(f"  Stability threshold = {args.stability_baseline * args.stability_threshold_factor:.4f} (max {args.stability_threshold_factor*100-100:.1f}% worse)")
    print()
    
    # Training loop
    print(f"\n[EDON-V8] Starting training: {args.episodes} episodes, profile={args.profile}")
    print("="*70)
    
    episode_scores = []
    episode_rewards = []
    episode_lengths = []
    episode_interventions = []
    episode_time_to_intervention = []
    episode_near_fail_density = []
    
    for episode in range(args.episodes):
        try:
            # Print episode start
            if episode == 0:
                print(f"[V8] Starting episode {episode + 1}/{args.episodes}...", flush=True)
            elif (episode + 1) % 10 == 0:
                print(f"[V8] Starting episode {episode + 1}/{args.episodes}...", flush=True)
            # Collect trajectory
            print(f"[V8] Collecting trajectory for episode {episode + 1}...", flush=True)
            trajectory = collect_trajectory(env, policy, max_steps=args.max_steps, device=device)
            print(f"[V8] Collected {len(trajectory['rewards'])} steps", flush=True)
        except Exception as e:
            print(f"[TRAIN] Error collecting trajectory: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise  # Re-raise to see the full error
        
        if len(trajectory["rewards"]) == 0:
            continue
        
        # Compute episode metrics for stability constraint
        episode_length = len(trajectory["rewards"])
        interventions = sum(1 for info in trajectory["infos"] if info.get("intervention", False))
        
        # RETROACTIVE INTERVENTION PENALTY (Credit Assignment)
        # Penalize the last N steps before each intervention to help the policy learn
        # that actions leading up to intervention are bad, not just the event itself
        # IMPORTANT: Apply this BEFORE computing returns/advantages so it affects learning
        retroactive_penalty_total = 0.0
        if args.retroactive_steps > 0 and args.w_retroactive > 0:
            # Find all intervention steps
            intervention_indices = []
            for i, info in enumerate(trajectory["infos"]):
                if info.get("intervention", False) or info.get("fallen", False):
                    intervention_indices.append(i)
            
            # Apply retroactive penalty to steps before each intervention
            for intervention_idx in intervention_indices:
                # Penalize the last retroactive_steps before this intervention
                start_idx = max(0, intervention_idx - args.retroactive_steps)
                end_idx = intervention_idx
                
                # Apply penalty to each step in this window
                # Use a decay factor: steps closer to intervention get higher penalty
                for step_idx in range(start_idx, end_idx):
                    # Decay: steps closer to intervention get more penalty
                    # Distance from intervention: (end_idx - step_idx)
                    # Closer steps (distance=1) get full penalty, farther steps get less
                    distance = end_idx - step_idx
                    decay_factor = 1.0 / distance  # Linear decay: 1.0, 0.5, 0.33, ...
                    
                    retroactive_penalty = args.w_retroactive * decay_factor
                    trajectory["rewards"][step_idx] -= retroactive_penalty
                    retroactive_penalty_total += retroactive_penalty
        
        # Store retroactive penalty for logging (use a simple variable, not env attribute)
        # This will be used later in the print statement
        
        # DIAGNOSTIC: Compute reward component breakdown
        # Recompute rewards with component breakdown for all steps
        from training.edon_score import step_reward
        reward_components_episode = {
            "intervention": 0.0,
            "stability_total": 0.0,
            "torque_total": 0.0,
            "alive_bonus": 0.0,
            "other": 0.0
        }
        
        # Use actual observation dicts if available, otherwise sample
        obs_dicts = trajectory.get("observations_dict", trajectory["observations"])
        
        # Sample every Nth step to avoid too much overhead (use all if < 50 steps)
        sample_rate = max(1, len(obs_dicts) // 50) if len(obs_dicts) > 50 else 1
        sample_indices = list(range(0, len(obs_dicts), sample_rate))
        if len(sample_indices) == 0:
            sample_indices = [0]
        
        for idx in sample_indices:
            obs_dict = obs_dicts[idx]
            info = trajectory["infos"][idx]
            prev_obs_dict = obs_dicts[idx - 1] if idx > 0 else None
            
            # Ensure obs_dict is a dict (not array)
            if isinstance(obs_dict, np.ndarray):
                # Skip if it's an observation vector (can't easily convert back)
                continue
            
            try:
                components = step_reward(
                    prev_state=prev_obs_dict,
                    next_state=obs_dict,
                    info=info,
                    w_intervention=args.w_intervention,
                    w_stability=args.w_stability,
                    w_torque=args.w_torque,
                    return_components=True
                )
                
                reward_components_episode["intervention"] += components["intervention"]
                reward_components_episode["stability_total"] += (
                    components["stability_tilt"] + 
                    components["stability_velocity"] + 
                    components["stability_oscillation"] + 
                    components["stability_phase_lag"]
                )
                reward_components_episode["torque_total"] += (
                    components["torque_action"] + 
                    components["torque_jerk"]
                )
                reward_components_episode["alive_bonus"] += components["alive_bonus"]
            except Exception as e:
                # Skip if reward computation fails
                continue
        
        # Scale up to full episode (approximate)
        if len(sample_indices) > 0:
            scale_factor = len(obs_dicts) / len(sample_indices)
            for key in reward_components_episode:
                reward_components_episode[key] *= scale_factor
        
        # Compute episode stability (average of stability scores from observations)
        episode_stability = 0.0
        stability_count = 0
        for info in trajectory["infos"]:
            if "stability_score" in info:
                episode_stability += info["stability_score"]
                stability_count += 1
        if stability_count > 0:
            episode_stability = episode_stability / stability_count
        else:
            # Fallback: estimate from tilt magnitudes
            tilt_sum = 0.0
            for obs in trajectory["observations"]:
                if isinstance(obs, dict):
                    roll = abs(obs.get("roll", 0.0))
                    pitch = abs(obs.get("pitch", 0.0))
                    tilt_sum += roll + pitch
            if len(trajectory["observations"]) > 0:
                episode_stability = tilt_sum / len(trajectory["observations"])
        
        # Apply episode-level stability constraint penalty BEFORE computing returns
        # This ensures the constraint affects learning
        stability_threshold = args.stability_baseline * args.stability_threshold_factor
        if episode_stability > stability_threshold:
            stability_excess = episode_stability - stability_threshold
            stability_penalty = args.w_stability_episode * stability_excess
            # Apply penalty evenly across all steps in trajectory
            per_step_penalty = stability_penalty / len(trajectory["rewards"])
            for i in range(len(trajectory["rewards"])):
                trajectory["rewards"][i] -= per_step_penalty
        
        # Compute returns (after applying stability constraint)
        returns = ppo.compute_returns(trajectory["rewards"], trajectory["dones"])
        
        # Compute advantages (simple baseline: mean return)
        # FIXED: Don't use GAE with returns as values (causes advantages to be zero)
        # Instead, use simple advantage = return - baseline
        mean_return = np.mean(returns)
        advantages = [r - mean_return for r in returns]
        
        # Normalize advantages (important for stable learning)
        if len(advantages) > 1:
            adv_mean = np.mean(advantages)
            adv_std = np.std(advantages)
            if adv_std > 1e-8:
                advantages = [(a - adv_mean) / adv_std for a in advantages]
        
        # Update policy
        update_metrics = ppo.update(
            obs_batch=trajectory["observations"],
            strategy_batch=trajectory["strategies"],
            modulations_batch=trajectory["modulations"],
            old_log_probs=trajectory["log_probs"],
            returns=returns,
            advantages=advantages,
            epochs=args.update_epochs
        )
        
        # Compute total reward (after stability penalty)
        total_reward = sum(trajectory["rewards"])
        
        # Compute v8 metrics
        episode_data = [
            {"obs": obs, "info": info, "done": done}
            for obs, info, done in zip(trajectory["observations"], trajectory["infos"], trajectory["dones"])
        ]
        v8_metrics = compute_episode_metrics_v8(episode_data)
        episode_score = v8_metrics.get("edon_score_v8", 0.0)
        time_to_intervention = v8_metrics.get("time_to_first_intervention")
        near_fail_density = v8_metrics.get("near_fail_density", 0.0)
        
        episode_scores.append(episode_score)
        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        episode_interventions.append(interventions)
        if time_to_intervention is not None:
            episode_time_to_intervention.append(time_to_intervention)
        episode_near_fail_density.append(near_fail_density)
        
        # Print progress
        time_str = f"{time_to_intervention:.0f}" if time_to_intervention is not None else "N/A"
        
        # Print reward component breakdown every 10 episodes or first episode
        reward_breakdown_str = ""
        if (episode + 1) % 10 == 0 or episode == 0:
            reward_breakdown_str = (
                f" | R_int={reward_components_episode['intervention']:.1f} "
                f"R_stab={reward_components_episode['stability_total']:.1f} "
                f"R_torque={reward_components_episode['torque_total']:.1f} "
                f"R_alive={reward_components_episode['alive_bonus']:.1f}"
            )
        
        print(
            f"[V8] ep={episode + 1}/{args.episodes} "
            f"score={episode_score:.2f} "
            f"reward={total_reward:.2f} "
            f"len={episode_length} "
            f"interventions={interventions} "
            f"time_to_int={time_str} "
            f"near_fail={near_fail_density:.4f} "
            f"policy_loss={update_metrics['policy_loss']:.4e} "
            f"entropy={update_metrics.get('entropy', 0.0):.4f} "
            f"kl={update_metrics.get('kl_div', 0.0):.4e}"
            f"{reward_breakdown_str}"
        )
        
        # Summary every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_score = np.mean(episode_scores[-10:])
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            avg_interventions = np.mean(episode_interventions[-10:])
            avg_time_to_int = np.mean(episode_time_to_intervention[-10:]) if len(episode_time_to_intervention) >= 10 else None
            avg_near_fail = np.mean(episode_near_fail_density[-10:])
            
            time_str = f"{avg_time_to_int:.1f}" if avg_time_to_int is not None else "N/A"
            print(
                f"\n[V8-SUMMARY] Episodes {episode + 1 - 9}-{episode + 1}: "
                f"avg_score={avg_score:.2f}, "
                f"avg_reward={avg_reward:.2f}, "
                f"avg_length={avg_length:.1f}, "
                f"avg_interventions={avg_interventions:.1f}, "
                f"avg_time_to_int={time_str}, "
                f"avg_near_fail={avg_near_fail:.4f}\n"
            )
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{args.model_name}.pt"
    
    print(f"\n[EDON-V8] Saving model to {model_path}")
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "input_size": obs_dim,
        "episodes": args.episodes,
        "final_avg_score": np.mean(episode_scores[-10:]) if len(episode_scores) >= 10 else np.mean(episode_scores)
    }, model_path)
    
    print(f"\n[EDON-V8] Training complete!")
    print("="*70)
    
    # Final summary (last 10 episodes)
    if len(episode_scores) >= 10:
        final_scores = episode_scores[-10:]
        final_rewards = episode_rewards[-10:]
        final_time_to_int = episode_time_to_intervention[-10:] if len(episode_time_to_intervention) >= 10 else []
        final_near_fail = episode_near_fail_density[-10:]
        
        print(f"Final Summary (last 10 episodes):")
        print(f"  Average score: {np.mean(final_scores):.2f}")
        print(f"  Average reward: {np.mean(final_rewards):.2f}")
        if final_time_to_int:
            print(f"  Average time-to-intervention: {np.mean(final_time_to_int):.1f} steps")
        print(f"  Average near-fail density: {np.mean(final_near_fail):.4f}")
    else:
        print(f"Final Summary (all {len(episode_scores)} episodes):")
        print(f"  Average score: {np.mean(episode_scores):.2f}")
        print(f"  Average reward: {np.mean(episode_rewards):.2f}")
        if episode_time_to_intervention:
            print(f"  Average time-to-intervention: {np.mean(episode_time_to_intervention):.1f} steps")
        print(f"  Average near-fail density: {np.mean(episode_near_fail_density):.4f}")
    
    print(f"\n  Model saved to: {model_path}")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL] Training failed with error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

