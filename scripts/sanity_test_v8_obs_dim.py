"""
Sanity test for v8 strategy policy with obs_dim inference.

Runs 1 episode and prints:
- obs_dim (inferred from environment)
- First Linear layer in_features
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.edon_humanoid_env import EdonHumanoidEnv
from env.edon_humanoid_env_v8 import EdonHumanoidEnvV8
from training.edon_v8_policy import EdonV8StrategyPolicy
from training.fail_risk_model import FailRiskModel
from run_eval import baseline_controller


def main():
    parser = argparse.ArgumentParser(description="Sanity test v8 obs_dim inference")
    parser.add_argument("--profile", type=str, default="high_stress", help="Stress profile")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fail-risk-model", type=str, default="models/edon_fail_risk_v1_fixed_v2.pt", help="Fail-risk model path")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per episode")
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create base environment
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
            print(f"[SANITY] Loaded fail-risk model from {args.fail_risk_model}")
    except Exception as e:
        print(f"[SANITY] Warning: Could not load fail-risk model: {e}")
    
    # Create a temporary dummy policy to create the env for obs_dim inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SANITY] Using device: {device}")
    
    dummy_policy = EdonV8StrategyPolicy(input_size=248).to(device)  # Temporary dummy
    temp_env = EdonHumanoidEnvV8(
        strategy_policy=dummy_policy,
        fail_risk_model=fail_risk_model,
        base_env=base_env,
        seed=args.seed,
        profile=args.profile,
        device=device
    )
    
    # Infer obs_dim from environment
    obs_dim = EdonV8StrategyPolicy._infer_obs_dim_from_env(temp_env)
    print(f"\n[SANITY] Inferred obs_dim from env: {obs_dim}")
    
    # Create policy with inferred obs_dim (NOT loading any old checkpoints)
    policy = EdonV8StrategyPolicy(input_size=obs_dim).to(device)
    
    # Print obs_dim and first layer in_features
    first_layer = policy.feature_net[0]
    print(f"[SANITY] obs_dim: {obs_dim}")
    print(f"[SANITY] First Linear layer in_features: {first_layer.in_features}")
    print(f"[SANITY] First Linear layer out_features: {first_layer.out_features}")
    
    # Recreate v8 environment with the correct policy
    env = EdonHumanoidEnvV8(
        strategy_policy=policy,
        fail_risk_model=fail_risk_model,
        base_env=base_env,
        seed=args.seed,
        profile=args.profile,
        device=device
    )
    
    # Run 1 episode
    print(f"\n[SANITY] Running 1 episode (max {args.max_steps} steps)...")
    obs = env.reset()
    step = 0
    total_reward = 0.0
    
    while step < args.max_steps:
        next_obs, reward, done, info = env.step(edon_core_state=None)
        total_reward += reward
        step += 1
        
        if step % 100 == 0:
            print(f"[SANITY] Step {step}/{args.max_steps}, reward={reward:.3f}, done={done}")
        
        if done:
            break
        
        obs = next_obs
    
    print(f"\n[SANITY] Episode complete!")
    print(f"[SANITY] Total steps: {step}")
    print(f"[SANITY] Total reward: {total_reward:.2f}")
    print(f"[SANITY] Final obs_dim: {obs_dim}")
    print(f"[SANITY] Final first layer in_features: {first_layer.in_features}")
    
    print("\n[SANITY] ✅ Sanity test passed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[SANITY] ❌ Sanity test failed with error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

