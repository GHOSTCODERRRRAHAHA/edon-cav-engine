"""
Reflex-Only Ablation Test

Tests baseline + reflex controller (no strategy layer) vs pure baseline.
If reflex alone regresses, it's not a safe base for learning.
"""

import sys
from pathlib import Path
import numpy as np
import json
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_eval import make_humanoid_env, baseline_controller
from controllers.edon_v8_reflex import EdonReflexController
from training.fail_risk_model import FailRiskModel
import torch


def run_reflex_only_eval(episodes: int = 30, seed: int = 42, profile: str = "high_stress"):
    """Run evaluation with baseline + reflex only (no strategy)."""
    print("="*80)
    print("Reflex-Only Ablation Test")
    print("="*80)
    print(f"Episodes: {episodes}, Seed: {seed}, Profile: {profile}")
    print()
    
    # Create environment
    base_env = make_humanoid_env(seed=seed, profile=profile)
    from env.edon_humanoid_env import EdonHumanoidEnv
    env = EdonHumanoidEnv(base_env=base_env, seed=seed, profile=profile)
    
    # Load fail-risk model (needed for reflex controller)
    fail_risk_model = None
    fail_risk_model_path = Path("models/edon_fail_risk_v1_fixed_v2.pt")
    if fail_risk_model_path.exists():
        checkpoint = torch.load(fail_risk_model_path, map_location="cpu", weights_only=False)
        input_size = checkpoint.get("input_size", 15)
        fail_risk_model = FailRiskModel(input_size=input_size)
        fail_risk_model.load_state_dict(checkpoint["model_state_dict"])
        fail_risk_model.eval()
        print(f"[ABLATION] Loaded fail-risk model from {fail_risk_model_path}")
    else:
        print("[ABLATION] Warning: Fail-risk model not found, using fail-risk=0.0")
    
    # Create reflex controller
    reflex_controller = EdonReflexController(
        max_damping_factor=1.3,
        fail_risk_damping_scale=1.5,
        tilt_damping_threshold=0.15,
        vel_damping_threshold=1.0
    )
    
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
                    # Extract features (simplified - would need full feature extraction)
                    features = np.array([
                        obs.get("roll", 0.0),
                        obs.get("pitch", 0.0),
                        obs.get("roll_velocity", 0.0),
                        obs.get("pitch_velocity", 0.0),
                        obs.get("com_velocity_x", 0.0),
                        obs.get("com_velocity_y", 0.0),
                        obs.get("com_velocity_z", 0.0),
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Placeholders
                    ])[:15]
                    with torch.no_grad():
                        fail_risk = float(torch.sigmoid(fail_risk_model(torch.FloatTensor(features).unsqueeze(0))).item())
                except:
                    pass
            
            fail_risks.append(fail_risk)
            
            # Apply reflex controller (no strategy modulations)
            final_action = reflex_controller.compute_action(
                baseline_action=baseline_action,
                obs=obs,
                fail_risk=fail_risk,
                strategy_modulations=None  # No strategy!
            )
            
            # Step environment
            next_obs, reward, done, info = env.step(final_action)
            
            # Track metrics
            if info.get("intervention", False):
                interventions += 1
            
            # Compute stability (tilt magnitude)
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
    print("Reflex-Only Results:")
    print(f"  Interventions/episode: {avg_interventions:.2f}")
    print(f"  Stability (avg): {avg_stability:.4f}")
    print(f"  Episode length (avg): {avg_length:.1f} steps")
    print(f"  Avg fail-risk: {avg_fail_risk:.4f}")
    print("="*80)
    
    # Save results
    output_file = Path("results/ablation_reflex_only.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            "mode": "reflex_only",
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
    
    run_reflex_only_eval(episodes=args.episodes, seed=args.seed, profile=args.profile)

