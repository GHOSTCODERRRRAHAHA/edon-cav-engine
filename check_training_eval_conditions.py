"""
Check Training vs Evaluation Conditions

Compare the exact conditions used in training vs evaluation to identify differences.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_eval import make_humanoid_env
from env.edon_humanoid_env import EdonHumanoidEnv

print("="*80)
print("Training vs Evaluation Conditions Comparison")
print("="*80)

# Training conditions (from train_edon_v8_strategy.py)
print("\n[1] TRAINING CONDITIONS:")
print("-"*80)
train_seed = 0
train_profile = "high_stress"
print(f"  Seed: {train_seed}")
print(f"  Profile: {train_profile}")
print(f"  Code: base_env = EdonHumanoidEnv(seed={train_seed}, profile='{train_profile}')")
print(f"        env = EdonHumanoidEnvV8(..., base_env=base_env, ...)")

# Create training-style environment
train_base = EdonHumanoidEnv(seed=train_seed, profile=train_profile)
train_obs = train_base.reset()
print(f"\n  Training base_env type: {type(train_base).__name__}")
print(f"  Training base_env.env type: {type(train_base.env).__name__}")
print(f"  Training obs keys: {list(train_obs.keys()) if isinstance(train_obs, dict) else 'N/A'}")

# Evaluation conditions (from eval_v8_memory_features.py)
print("\n[2] EVALUATION CONDITIONS:")
print("-"*80)
eval_seed = 42
eval_profile = "high_stress"
print(f"  Seed: {eval_seed}")
print(f"  Profile: {eval_profile}")
print(f"  Code: base_env = make_humanoid_env(seed={eval_seed}, profile='{eval_profile}')")
print(f"        env = EdonHumanoidEnvV8(..., base_env=base_env, ...)")

# Create evaluation-style environment
eval_base = make_humanoid_env(seed=eval_seed, profile=eval_profile)
eval_obs = eval_base.reset()
print(f"\n  Evaluation base_env type: {type(eval_base).__name__}")
print(f"  Evaluation obs keys: {list(eval_obs.keys()) if isinstance(eval_obs, dict) else 'N/A'}")

# Check differences
print("\n[3] KEY DIFFERENCES:")
print("-"*80)

# Difference 1: Seed
if train_seed != eval_seed:
    print(f"  ⚠️  SEED MISMATCH:")
    print(f"     Training: {train_seed}")
    print(f"     Evaluation: {eval_seed}")
    print(f"     Impact: Different random initializations = different episodes!")

# Difference 2: Environment wrapper
if type(train_base).__name__ != type(eval_base).__name__:
    print(f"\n  ⚠️  ENVIRONMENT WRAPPER DIFFERENCE:")
    print(f"     Training base_env: {type(train_base).__name__}")
    print(f"     Evaluation base_env: {type(eval_base).__name__}")
    print(f"     Impact: Training wraps MockHumanoidEnv in EdonHumanoidEnv first,")
    print(f"             Evaluation passes MockHumanoidEnv directly to EdonHumanoidEnvV8")
    print(f"     This could cause different behavior!")

# Difference 3: Profile
if train_profile != eval_profile:
    print(f"\n  ⚠️  PROFILE MISMATCH:")
    print(f"     Training: {train_profile}")
    print(f"     Evaluation: {eval_profile}")
else:
    print(f"  ✓ Profile matches: {train_profile}")

# Check if EdonHumanoidEnv wrapper changes behavior
print("\n[4] ENVIRONMENT WRAPPER ANALYSIS:")
print("-"*80)
print(f"  Training creates: EdonHumanoidEnvV8 -> EdonHumanoidEnv -> MockHumanoidEnv")
print(f"  Evaluation creates: EdonHumanoidEnvV8 -> MockHumanoidEnv")
print(f"  ")
print(f"  The EdonHumanoidEnv wrapper in training adds:")
print(f"    - EDON reward computation (step_reward)")
print(f"    - Reward weights (w_intervention, w_stability, w_torque)")
print(f"    - Episode counting")
print(f"  ")
print(f"  But EdonHumanoidEnvV8 should handle rewards itself, so this might be redundant!")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)
print("1. Re-run evaluation with seed=0 to match training")
print("2. Consider removing EdonHumanoidEnv wrapper from training")
print("   (use make_humanoid_env directly like evaluation does)")
print("="*80)

