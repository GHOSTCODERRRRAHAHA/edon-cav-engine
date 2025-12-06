#!/usr/bin/env python
"""Quick test to verify obs_dim inference works"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from training.edon_v8_policy import EdonV8StrategyPolicy
    print("✓ Policy import successful")
    
    from env.edon_humanoid_env import EdonHumanoidEnv
    print("✓ Env import successful")
    
    from env.edon_humanoid_env_v8 import EdonHumanoidEnvV8
    print("✓ V8 Env import successful")
    
    # Create base env
    base_env = EdonHumanoidEnv(seed=42, profile="high_stress")
    print("✓ Base env created")
    
    # Create dummy policy for temp env
    import torch
    device = "cpu"
    dummy_policy = EdonV8StrategyPolicy(input_size=248).to(device)
    print("✓ Dummy policy created")
    
    # Create temp env
    temp_env = EdonHumanoidEnvV8(
        strategy_policy=dummy_policy,
        fail_risk_model=None,
        base_env=base_env,
        seed=42,
        profile="high_stress",
        device=device
    )
    print("✓ Temp env created")
    
    # Infer obs_dim
    obs_dim = EdonV8StrategyPolicy._infer_obs_dim_from_env(temp_env)
    print(f"✓ Inferred obs_dim: {obs_dim}")
    
    # Create policy with inferred obs_dim
    policy = EdonV8StrategyPolicy(input_size=obs_dim).to(device)
    print("✓ Policy created with inferred obs_dim")
    
    # Print first layer info
    first_layer = policy.feature_net[0]
    print(f"✓ obs_dim: {obs_dim}")
    print(f"✓ First Linear layer in_features: {first_layer.in_features}")
    print(f"✓ First Linear layer out_features: {first_layer.out_features}")
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

