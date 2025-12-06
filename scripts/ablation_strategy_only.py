"""
Strategy-Only Ablation Test

Tests strategy layer with minimal reflex (just clipping/very light damping).
If strategy can't find positive direction, reward & fail-risk alignment is the problem.
"""

import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import json
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_eval import make_humanoid_env, baseline_controller
from training.edon_v8_policy import EdonV8StrategyPolicy, pack_observation_v8
from training.fail_risk_model import FailRiskModel


class MinimalReflexController:
    """Minimal reflex controller - just clipping and very light damping."""
    
    def compute_action(
        self,
        baseline_action: np.ndarray,
        obs: Dict[str, Any],
        fail_risk: float = 0.0,
        strategy_modulations: Dict[str, float] = None
    ) -> np.ndarray:
        """Apply minimal adjustments - mostly just strategy modulations."""
        action = baseline_action.copy()
        
        # Very light damping only for extreme cases
        roll = abs(obs.get("roll", 0.0))
        pitch = abs(obs.get("pitch", 0.0))
        tilt_mag = roll + pitch
        
        if tilt_mag > 0.4:  # Only for very high tilt
            action = action * 0.9  # Very light damping
        
        # Apply strategy modulations (this is the main control)
        if strategy_modulations:
            gain_scale = strategy_modulations.get("gain_scale", 1.0)
            action = action * gain_scale
            
            lateral_compliance = strategy_modulations.get("lateral_compliance", 1.0)
            if len(action) >= 4:
                action[:4] = action[:4] * lateral_compliance
            
            step_height_bias = strategy_modulations.get("step_height_bias", 0.0)
            if len(action) >= 8:
                action[4:8] = action[4:8] + step_height_bias * 0.1
        
        # Just clip to valid range
        action = np.clip(action, -1.0, 1.0)
        
        return action


def run_strategy_only_eval(episodes: int = 30, seed: int = 42, profile: str = "high_stress"):
    """Run evaluation with strategy + minimal reflex."""
    print("="*80)
    print("Strategy-Only Ablation Test")
    print("="*80)
    print(f"Episodes: {episodes}, Seed: {seed}, Profile: {profile}")
    print()
    
    # Create environment
    base_env = make_humanoid_env(seed=seed, profile=profile)
    from env.edon_humanoid_env import EdonHumanoidEnv
    env = EdonHumanoidEnv(base_env=base_env, seed=seed, profile=profile)
    
    # Load fail-risk model
    fail_risk_model = None
    fail_risk_model_path = Path("models/edon_fail_risk_v1_fixed_v2.pt")
    if fail_risk_model_path.exists():
        checkpoint = torch.load(fail_risk_model_path, map_location="cpu", weights_only=False)
        input_size = checkpoint.get("input_size", 15)
        fail_risk_model = FailRiskModel(input_size=input_size)
        fail_risk_model.load_state_dict(checkpoint["model_state_dict"])
        fail_risk_model.eval()
        print(f"[ABLATION] Loaded fail-risk model from {fail_risk_model_path}")
    
    # Load strategy policy
    strategy_policy = None
    strategy_model_path = Path("models/edon_v8_strategy_v1_fixed.pt")
    if strategy_model_path.exists():
        checkpoint = torch.load(strategy_model_path, map_location="cpu", weights_only=False)
        input_size = checkpoint.get("input_size", 25)
        strategy_policy = EdonV8StrategyPolicy(input_size=input_size)
        # Try different checkpoint formats
        if "model_state_dict" in checkpoint:
            strategy_policy.load_state_dict(checkpoint["model_state_dict"])
        elif "policy_state_dict" in checkpoint:
            strategy_policy.load_state_dict(checkpoint["policy_state_dict"])
        else:
            # Assume checkpoint is the state dict itself
            strategy_policy.load_state_dict(checkpoint)
        strategy_policy.eval()
        print(f"[ABLATION] Loaded strategy policy from {strategy_model_path}")
    else:
        print("[ABLATION] Error: Strategy policy not found!")
        return None
    
    # Create minimal reflex controller
    minimal_reflex = MinimalReflexController()
    
    # Run episodes
    episode_results = []
    
    for episode_id in range(episodes):
        obs = env.reset()
        done = False
        step_count = 0
        interventions = 0
        stability_scores = []
        fail_risks = []
        
        while not done and step_count < 1000:
            # Get baseline action
            baseline_action = baseline_controller(obs, edon_state=None)
            baseline_action = np.array(baseline_action)
            
            # Compute fail-risk
            fail_risk = 0.0
            if fail_risk_model:
                try:
                    features = np.array([
                        obs.get("roll", 0.0),
                        obs.get("pitch", 0.0),
                        obs.get("roll_velocity", 0.0),
                        obs.get("pitch_velocity", 0.0),
                        obs.get("com_velocity_x", 0.0),
                        obs.get("com_velocity_y", 0.0),
                        obs.get("com_velocity_z", 0.0),
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                    ])[:15]
                    with torch.no_grad():
                        fail_risk = float(torch.sigmoid(fail_risk_model(torch.FloatTensor(features).unsqueeze(0))).item())
                except:
                    pass
            
            fail_risks.append(fail_risk)
            
            # Get strategy from policy
            obs_vec = pack_observation_v8(
                obs=obs,
                baseline_action=baseline_action,
                fail_risk=fail_risk,
                instability_score=0.0,
                phase="stable"
            )
            
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs_vec).unsqueeze(0)
                strategy_id, modulations, _ = strategy_policy.sample_action(obs_tensor, deterministic=False)
            
            # Apply minimal reflex + strategy
            final_action = minimal_reflex.compute_action(
                baseline_action=baseline_action,
                obs=obs,
                fail_risk=fail_risk,
                strategy_modulations=modulations
            )
            
            # Step environment
            next_obs, reward, done, info = env.step(final_action)
            
            # Track metrics
            if info.get("intervention", False):
                interventions += 1
            
            roll = abs(obs.get("roll", 0.0))
            pitch = abs(obs.get("pitch", 0.0))
            stability_scores.append(roll + pitch)
            
            obs = next_obs
            step_count += 1
        
        # Episode summary
        avg_stability = np.mean(stability_scores) if stability_scores else 0.0
        avg_fail_risk = np.mean(fail_risks) if fail_risks else 0.0
        
        episode_results.append({
            "episode_id": episode_id,
            "interventions": interventions,
            "stability_score": avg_stability,
            "episode_length": step_count,
            "avg_fail_risk": avg_fail_risk
        })
        
        print(f"Episode {episode_id+1}/{episodes}... Interventions: {interventions}, Stability: {avg_stability:.4f}")
    
    # Summary
    avg_interventions = np.mean([e["interventions"] for e in episode_results])
    avg_stability = np.mean([e["stability_score"] for e in episode_results])
    avg_length = np.mean([e["episode_length"] for e in episode_results])
    avg_fail_risk = np.mean([e["avg_fail_risk"] for e in episode_results])
    
    print()
    print("="*80)
    print("Strategy-Only Results:")
    print(f"  Interventions/episode: {avg_interventions:.2f}")
    print(f"  Stability (avg): {avg_stability:.4f}")
    print(f"  Episode length (avg): {avg_length:.1f} steps")
    print(f"  Avg fail-risk: {avg_fail_risk:.4f}")
    print("="*80)
    
    # Save results
    output_file = Path("results/ablation_strategy_only.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            "mode": "strategy_only",
            "profile": profile,
            "episodes": episode_results,
            "summary": {
                "interventions_per_episode": avg_interventions,
                "stability_avg": avg_stability,
                "avg_episode_length": avg_length,
                "avg_fail_risk": avg_fail_risk
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return {
        "interventions_per_episode": avg_interventions,
        "stability_avg": avg_stability,
        "avg_episode_length": avg_length,
        "avg_fail_risk": avg_fail_risk
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", type=str, default="high_stress")
    args = parser.parse_args()
    
    run_strategy_only_eval(episodes=args.episodes, seed=args.seed, profile=args.profile)

