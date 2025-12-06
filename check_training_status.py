"""Quick script to check if v8 training is running and verify setup"""
import sys
from pathlib import Path
import time

print("=" * 70)
print("EDON v8 Training Status Check")
print("=" * 70)

# Check if model file exists
model_path = Path("models/edon_v8_strategy_memory_features.pt")
if model_path.exists():
    file_info = model_path.stat()
    size_mb = file_info.st_size / (1024 * 1024)
    mod_time = time.ctime(file_info.st_mtime)
    print(f"\n✓ Model file exists: {model_path}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Last modified: {mod_time}")
    print(f"  Status: Training may be complete or in progress")
else:
    print(f"\n⚠ Model file not found: {model_path}")
    print(f"  Status: Training may not have started yet, or is still running")

# Check Python processes (basic check)
print("\n" + "=" * 70)
print("To check if training is running:")
print("  Run: Get-Process python | Select-Object Id, CPU, StartTime")
print("=" * 70)

# Verify training setup
print("\nVerifying training setup...")
try:
    sys.path.insert(0, '.')
    from training.edon_v8_policy import EdonV8StrategyPolicy
    from env.edon_humanoid_env import EdonHumanoidEnv
    from env.edon_humanoid_env_v8 import EdonHumanoidEnvV8
    import torch
    
    print("✓ All imports successful")
    
    # Quick inference test
    base_env = EdonHumanoidEnv(seed=0, profile="high_stress")
    device = "cpu"
    dummy_policy = EdonV8StrategyPolicy(input_size=248).to(device)
    temp_env = EdonHumanoidEnvV8(
        strategy_policy=dummy_policy,
        fail_risk_model=None,
        base_env=base_env,
        seed=0,
        profile="high_stress",
        device=device
    )
    obs_dim = EdonV8StrategyPolicy._infer_obs_dim_from_env(temp_env)
    print(f"✓ obs_dim inference works: {obs_dim}")
    print("✓ Training setup is correct")
    
except Exception as e:
    print(f"✗ Setup verification failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Training Command:")
print("  python training/train_edon_v8_strategy.py \\")
print("    --episodes 300 \\")
print("    --profile high_stress \\")
print("    --model-name edon_v8_strategy_memory_features")
print("=" * 70)

