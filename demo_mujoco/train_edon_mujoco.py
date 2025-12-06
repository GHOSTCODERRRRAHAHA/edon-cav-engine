"""
EDON Training for MuJoCo Environment using OEM API

This script trains EDON's policy network on the MuJoCo environment using
the same OEM API endpoints that customers will use in production.

Training Process:
1. Uses MuJoCo HumanoidEnv (same as demo)
2. Calls OEM API: POST /oem/robot/stability (for inference)
3. Records outcomes: POST /oem/robot/stability/record-outcome (for learning)
4. Trains policy network using PPO (same as original training)
5. Saves trained model for use in demo

This demonstrates the full OEM training workflow.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import requests
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.env import HumanoidEnv
from controllers.baseline_controller import BaselineController
from disturbances.generator import DisturbanceGenerator
from metrics.tracker import MetricsTracker
from training.edon_v8_policy import EdonV8StrategyPolicy, pack_stacked_observation_v8
from training.fail_risk_model import FailRiskModel

# Import stress profile
try:
    from evaluation.stress_profiles import get_stress_profile, HIGH_STRESS
    STRESS_PROFILE_AVAILABLE = True
except ImportError:
    STRESS_PROFILE_AVAILABLE = False


class PPO:
    """PPO algorithm for v8 strategy policy."""
    
    def __init__(
        self,
        policy: EdonV8StrategyPolicy,
        lr: float = 5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.02
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
        """Compute Generalized Advantage Estimation (GAE)."""
        if values is None or len(values) != len(rewards):
            returns_list = self.compute_returns(rewards, dones or [False] * len(rewards), next_value)
            values = returns_list
        
        advantages = []
        gae = 0.0
        for step in reversed(range(len(rewards))):
            if dones and dones[step]:
                gae = 0.0
            delta = rewards[step] + self.gamma * (next_value if step == len(rewards) - 1 else values[step + 1]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            next_value = values[step]
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
    
    def update(
        self,
        observations: List[np.ndarray],
        actions: List[Tuple[int, Dict[str, float]]],  # (strategy_id, modulations)
        rewards: List[float],
        dones: List[bool],
        old_log_probs: List[float],
        update_epochs: int = 10
    ):
        """Update policy using PPO."""
        if len(observations) == 0:
            return {}
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(observations))
        strategy_ids = torch.LongTensor([a[0] for a in actions])
        modulations_list = [a[1] for a in actions]
        
        # Compute advantages
        advantages, returns = self.compute_gae(rewards, dones=dones)
        advantages_tensor = torch.FloatTensor(advantages)
        returns_tensor = torch.FloatTensor(returns)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        old_log_probs_tensor = torch.FloatTensor(old_log_probs)
        
        total_loss = 0.0
        for epoch in range(update_epochs):
            # Forward pass - policy returns (strategy_logits, modulations_dict)
            strategy_logits, modulations_dict = self.policy(obs_tensor)
            
            # Strategy distribution
            strategy_dist = Categorical(logits=strategy_logits)
            strategy_log_probs = strategy_dist.log_prob(strategy_ids)
            
            # Compute ratios
            ratio = torch.exp(strategy_log_probs - old_log_probs_tensor)
            
            # PPO clipped objective
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Entropy bonus
            entropy = strategy_dist.entropy().mean()
            entropy_loss = -self.entropy_coef * entropy
            
            # Total loss
            loss = policy_loss + entropy_loss
            total_loss += loss.item()
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        return {
            "loss": total_loss / update_epochs,
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "mean_advantage": advantages_tensor.mean().item()
        }


class MuJoCoTrainingEnv:
    """
    Wrapper around MuJoCo HumanoidEnv for training.
    
    Uses OEM API endpoints for EDON inference and outcome recording.
    """
    
    def __init__(
        self,
        edon_base_url: str = "http://localhost:8000",
        stress_profile=None,
        dt: float = 0.01
    ):
        self.edon_base_url = edon_base_url
        self.dt = dt
        
        # Create MuJoCo environment
        if STRESS_PROFILE_AVAILABLE and stress_profile:
            self.env = HumanoidEnv(
                dt=dt,
                render=False,
                sensor_noise_std=stress_profile.sensor_noise_std,
                actuator_delay_steps=stress_profile.actuator_delay_steps,
                friction_min=stress_profile.friction_min,
                friction_max=stress_profile.friction_max,
                fatigue_enabled=stress_profile.fatigue_enabled,
                fatigue_degradation=stress_profile.fatigue_degradation,
                floor_incline_range=stress_profile.floor_incline_range,
                height_variation_range=stress_profile.height_variation_range
            )
        else:
            self.env = HumanoidEnv(dt=dt, render=False)
        
        self.baseline_controller = BaselineController()
        self.disturbance_generator = DisturbanceGenerator()
        self.metrics = MetricsTracker()
        
        # Observation history for stacked observations
        self.obs_history = deque(maxlen=8)
        
    def reset(self, seed: Optional[int] = None, disturbance_script: Optional[List] = None):
        """Reset environment."""
        obs, info = self.env.reset(seed=seed, disturbance_script=disturbance_script)
        self.obs_history.clear()
        for _ in range(8):
            self.obs_history.append(obs)
        return obs, info
    
    def step(self, action: np.ndarray):
        """Step environment."""
        # HumanoidEnv.step() returns (obs, done, info) - no reward
        obs, done, info = self.env.step(action)
        self.obs_history.append(obs)
        # Return with reward=0.0 (reward is computed in collect_trajectory)
        return obs, 0.0, done, info
    
    def get_stacked_observation(self, baseline_action: Optional[np.ndarray] = None, fail_risk: float = 0.5) -> np.ndarray:
        """
        Get stacked observation (last 8 frames).
        
        Args:
            baseline_action: Baseline action for packing (if None, uses zeros)
            fail_risk: Current fail risk (from EDON API)
        """
        if baseline_action is None:
            baseline_action = np.zeros(12)  # Default dummy baseline
        
        # Get current observation (last in history)
        current_obs = self.obs_history[-1] if len(self.obs_history) > 0 else {}
        
        # Pack stacked observation with history
        return pack_stacked_observation_v8(
            obs=current_obs,
            baseline_action=baseline_action,
            fail_risk=fail_risk,
            instability_score=0.0,
            phase="stable",
            obs_history=list(self.obs_history),
            near_fail_history=[],
            obs_vec_history=[],  # Will be built up over time
            stack_size=8
        )
    
    def call_edon_api(self, robot_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call EDON API: POST /oem/robot/stability
        
        Note: This is the EDON API endpoint (path: /oem/robot/stability).
        This is the same API endpoint OEMs will use in production.
        """
        url = f"{self.edon_base_url}/oem/robot/stability"
        try:
            response = requests.post(
                url,
                json={"robot_state": robot_state},
                timeout=2.0
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Warning: EDON API call failed: {e}")
            # Fallback to default
            return {
                "strategy_id": 0,
                "strategy_name": "NORMAL",
                "modulations": {"gain_scale": 1.0, "lateral_compliance": 1.0, "step_height_bias": 0.0},
                "fail_risk": 0.5
            }
    
    def record_outcome(
        self,
        strategy_id: int,
        modulations: Dict[str, float],
        fail_risk: float,
        robot_state: Dict[str, Any],
        intervention_occurred: bool
    ):
        """
        Record intervention outcome: POST /oem/robot/stability/record-outcome
        
        Note: This is the EDON API endpoint (path: /oem/robot/stability/record-outcome).
        This is the same API endpoint OEMs will use for adaptive learning.
        """
        url = f"{self.edon_base_url}/oem/robot/stability/record-outcome"
        try:
            # Extract modulations to match RecordOutcomeRequest model
            gain_scale = modulations.get("gain_scale", 1.0)
            lateral_compliance = modulations.get("lateral_compliance", 1.0)
            step_height_bias = modulations.get("step_height_bias", 0.0)
            
            # Build request body matching RecordOutcomeRequest model
            request_body = {
                "strategy_id": strategy_id,
                "gain_scale": gain_scale,
                "lateral_compliance": lateral_compliance,
                "step_height_bias": step_height_bias,
                "fail_risk": fail_risk,
                "robot_state": robot_state,
                "intervention_occurred": intervention_occurred
            }
            
            requests.post(
                url,
                json=request_body,
                timeout=0.05  # Non-blocking
            )
        except Exception:
            pass  # Non-blocking, ignore errors


def collect_trajectory(
    env: MuJoCoTrainingEnv,
    policy: EdonV8StrategyPolicy,
    max_steps: int = 1000,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Collect a trajectory using policy network and OEM API.
    
    The policy network learns strategy selection, while EDON Core API
    provides fail-risk and base modulations.
    """
    obs, info = env.reset()
    done = False
    step_count = 0
    
    observations = []
    actions = []  # (strategy_id, modulations)
    rewards = []
    dones = []
    old_log_probs = []
    infos = []  # Store info dicts for intervention tracking
    
    # Track for outcome recording
    last_edon_response = None
    last_robot_state = None
    
    while not done and step_count < max_steps:
        # Get baseline action
        baseline_action = env.baseline_controller.step(obs)
        
        # Call EDON API (OEM endpoint)
        robot_state = {
            "roll": float(obs.get("roll", 0.0)),
            "pitch": float(obs.get("pitch", 0.0)),
            "roll_velocity": float(obs.get("roll_velocity", 0.0)),
            "pitch_velocity": float(obs.get("pitch_velocity", 0.0)),
            "com_x": float(obs.get("com_x", 0.0)),
            "com_y": float(obs.get("com_y", 0.0))
        }
        edon_response = env.call_edon_api(robot_state)
        last_edon_response = edon_response
        last_robot_state = robot_state
        fail_risk = edon_response.get("fail_risk", 0.5)
        
        # Get stacked observation (with baseline action and fail_risk)
        stacked_obs = env.get_stacked_observation(baseline_action=baseline_action, fail_risk=fail_risk)
        observations.append(stacked_obs)
        
        # Policy network selects strategy and modulations
        obs_tensor = torch.FloatTensor(stacked_obs).unsqueeze(0).to(device)
        with torch.no_grad():
            strategy_logits, modulations_dict = policy(obs_tensor)
            strategy_dist = Categorical(logits=strategy_logits)
            strategy_id = strategy_dist.sample().item()
            strategy_log_prob = strategy_dist.log_prob(torch.LongTensor([strategy_id])).item()
        
        # Extract modulations from dict (already in correct ranges from policy)
        gain_scale = modulations_dict["gain_scale"].item() if modulations_dict["gain_scale"].numel() == 1 else modulations_dict["gain_scale"].squeeze().item()
        lateral_compliance = modulations_dict["lateral_compliance"].item() if modulations_dict["lateral_compliance"].numel() == 1 else modulations_dict["lateral_compliance"].squeeze().item()
        step_height_bias = modulations_dict["step_height_bias"].item() if modulations_dict["step_height_bias"].numel() == 1 else modulations_dict["step_height_bias"].squeeze().item()
        
        # Use policy's modulations (can override EDON's base modulations)
        modulations = {
            "gain_scale": float(gain_scale),
            "lateral_compliance": float(lateral_compliance),
            "step_height_bias": float(step_height_bias)
        }
        
        actions.append((strategy_id, modulations))
        old_log_probs.append(strategy_log_prob)
        
        # Apply modulations to baseline action
        # Normalize baseline action to [-1, 1] range
        baseline_normalized = np.clip(baseline_action / 20.0, -1.0, 1.0)
        
        # Apply gain scale
        corrected_action = baseline_normalized * modulations["gain_scale"]
        
        # Apply lateral_compliance to root rotation (indices 3-5: roll/pitch/yaw)
        # This matches the original training architecture, adapted for MuJoCo's action space
        if len(corrected_action) >= 6:
            corrected_action[3:6] = corrected_action[3:6] * modulations["lateral_compliance"]
        
        # Apply step_height_bias to leg joints (indices 6-11: legs)
        # This matches the original training architecture, adapted for MuJoCo's action space
        if len(corrected_action) >= 12:
            corrected_action[6:12] = corrected_action[6:12] + modulations["step_height_bias"] * 0.1
        
        # Clamp back to [-1, 1] and scale to MuJoCo range
        corrected_action = np.clip(corrected_action, -1.0, 1.0) * 20.0
        
        # Step environment
        obs, reward, done, info = env.step(corrected_action)
        
        # Check for intervention
        intervention = info.get("intervention_detected", False)
        
        # Record outcome (OEM API endpoint)
        if last_edon_response:
            env.record_outcome(
                strategy_id=last_edon_response.get("strategy_id", strategy_id),
                modulations=last_edon_response.get("modulations", modulations),
                fail_risk=last_edon_response.get("fail_risk", 0.5),
                robot_state=last_robot_state,
                intervention_occurred=intervention
            )
        
        # Compute reward
        # Reward structure: intervention penalty + stability bonus
        stability_reward = -abs(obs.get("roll", 0.0)) - abs(obs.get("pitch", 0.0))
        intervention_penalty = -20.0 if intervention else 0.0
        reward = stability_reward + intervention_penalty
        
        rewards.append(reward)
        dones.append(done)
        infos.append(info)  # Store info for intervention tracking
        step_count += 1
    
    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "old_log_probs": old_log_probs,
        "infos": infos,  # Include infos for intervention tracking
        "steps": step_count
    }


def main():
    parser = argparse.ArgumentParser(description="Train EDON on MuJoCo using OEM API")
    parser.add_argument("--episodes", type=int, default=300, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--update-epochs", type=int, default=10, help="PPO update epochs")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    parser.add_argument("--model-name", type=str, default="edon_v8_mujoco", help="Model name")
    parser.add_argument("--edon-url", type=str, default="http://localhost:8000", help="EDON API URL")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Enable adaptive memory for training (learns from outcomes)
    # This allows EDON to adapt during training, improving performance
    # For training, we want adaptive memory ON by default
    # Note: The EDON server process needs to have adaptive memory enabled
    # If you see warnings in server logs, restart the server without EDON_DISABLE_ADAPTIVE_MEMORY=1
    
    # Check if adaptive memory is disabled in this process
    if os.getenv("EDON_DISABLE_ADAPTIVE_MEMORY") == "1":
        print("⚠️  Note: Adaptive memory was disabled in this process")
        print("   Enabling for training script...")
        os.environ["EDON_DISABLE_ADAPTIVE_MEMORY"] = "0"
    
    # Ensure it's enabled for this process
    os.environ["EDON_DISABLE_ADAPTIVE_MEMORY"] = "0"
    print("✅ Adaptive memory ENABLED for training")
    print("   Note: Make sure your EDON server was started WITHOUT EDON_DISABLE_ADAPTIVE_MEMORY=1")
    print("   If unsure, restart server: python -m app.main (without the env var)")
    
    # Get stress profile
    if STRESS_PROFILE_AVAILABLE:
        stress_profile = HIGH_STRESS
    else:
        stress_profile = None
    
    # Create environment
    print("Creating MuJoCo training environment...")
    env = MuJoCoTrainingEnv(
        edon_base_url=args.edon_url,
        stress_profile=stress_profile,
        dt=0.01
    )
    
    # Create policy network
    print("Creating policy network...")
    # Infer input size from environment
    obs, _ = env.reset()
    stacked_obs = env.get_stacked_observation()
    input_size = len(stacked_obs)
    
    policy = EdonV8StrategyPolicy(input_size=input_size).to(args.device)
    
    # Create PPO trainer
    trainer = PPO(
        policy=policy,
        lr=args.lr,
        gamma=args.gamma,
        entropy_coef=0.02
    )
    
    print(f"\nStarting training with {args.episodes} episodes...")
    print(f"Using EDON API: {args.edon_url}/oem/robot/stability")
    print(f"Output: {args.output_dir}/{args.model_name}.pt\n")
    
    # Training loop with live updates
    episode_rewards = []
    episode_interventions = []
    episode_steps = []
    
    print("\n" + "="*70)
    print("TRAINING PROGRESS")
    print("="*70)
    print(f"{'Episode':<10} {'Steps':<8} {'Reward':<12} {'Interventions':<15} {'Loss':<10} {'Status'}")
    print("-"*70)
    
    for episode in range(args.episodes):
        # Collect trajectory
        trajectory = collect_trajectory(env, policy, max_steps=args.max_steps, device=args.device)
        
        if len(trajectory["observations"]) > 0:
            # Update policy
            update_info = trainer.update(
                observations=trajectory["observations"],
                actions=trajectory["actions"],
                rewards=trajectory["rewards"],
                dones=trajectory["dones"],
                old_log_probs=trajectory["old_log_probs"],
                update_epochs=args.update_epochs
            )
            
            # Track metrics
            total_reward = sum(trajectory["rewards"])
            episode_rewards.append(total_reward)
            
            # Count interventions (from info dicts, not dones)
            interventions = sum(1 for info in trajectory.get("infos", []) if info.get("intervention_detected", False))
            episode_interventions.append(interventions)
            
            # Track steps
            steps = len(trajectory["observations"])
            episode_steps.append(steps)
            
            # Live update every episode
            avg_reward_10 = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else total_reward
            avg_interventions_10 = np.mean(episode_interventions[-10:]) if len(episode_interventions) >= 10 else interventions
            loss = update_info.get('loss', 0.0)
            
            # Status indicator
            if interventions == 0:
                status = "✅ Stable"
            elif interventions <= 2:
                status = "⚠️  Minor"
            else:
                status = "❌ Unstable"
            
            # Print live update
            print(
                f"{episode + 1:<10} "
                f"{steps:<8} "
                f"{total_reward:>8.2f} ({avg_reward_10:>6.2f}) "
                f"{interventions:<4} (avg: {avg_interventions_10:<5.1f}) "
                f"{loss:>8.4f} "
                f"{status}"
            )
            
            # Print detailed stats every 10 episodes
            if (episode + 1) % 10 == 0:
                print("-"*70)
                print(f"Last 10 episodes: Avg reward={avg_reward_10:.2f}, "
                      f"Avg interventions={avg_interventions_10:.1f}, "
                      f"Avg steps={np.mean(episode_steps[-10:]):.1f}")
                print("-"*70)
        else:
            # Empty trajectory
            print(f"{episode + 1:<10} {'0':<8} {'N/A':<12} {'N/A':<15} {'N/A':<10} {'⚠️  Empty'}")
        
        # Save checkpoint
        if (episode + 1) % 50 == 0:
            checkpoint_path = os.path.join(args.output_dir, f"{args.model_name}_ep{episode+1}.pt")
            torch.save(policy.state_dict(), checkpoint_path)
            print(f"\n💾 Saved checkpoint: {checkpoint_path}\n")
    
    # Save final model
    final_path = os.path.join(args.output_dir, f"{args.model_name}.pt")
    torch.save(policy.state_dict(), final_path)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Saved model: {final_path}")
    print(f"\nPerformance Summary:")
    if len(episode_rewards) >= 10:
        print(f"  First 10 episodes - Avg reward: {np.mean(episode_rewards[:10]):.2f}, "
              f"Avg interventions: {np.mean(episode_interventions[:10]):.1f}")
        print(f"  Last 10 episodes  - Avg reward: {np.mean(episode_rewards[-10:]):.2f}, "
              f"Avg interventions: {np.mean(episode_interventions[-10:]):.1f}")
        improvement = np.mean(episode_interventions[:10]) - np.mean(episode_interventions[-10:])
        if improvement > 0:
            print(f"  Intervention reduction: {improvement:.1f} interventions/episode")
    print(f"\nTo use trained model in demo:")
    print(f"  python run_demo.py --mode trained --trained-model {final_path}")


if __name__ == "__main__":
    main()

