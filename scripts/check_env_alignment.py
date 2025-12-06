"""
Check Training vs Evaluation Environment Alignment

This script checks if the training and evaluation environments are set up the same way.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_eval import make_humanoid_env, baseline_controller
from env.edon_humanoid_env import EdonHumanoidEnv


def check_env_alignment():
    """Check if training and evaluation environments match."""
    print("="*80)
    print("Training vs Evaluation Environment Alignment Check")
    print("="*80)
    
    # Training environment (as used in train_edon_v8_strategy.py)
    print("\n[1] Training Environment Setup:")
    print("-"*80)
    print("Code: base_env = EdonHumanoidEnv(seed=args.seed, profile=args.profile)")
    print("      env = EdonHumanoidEnvV8(..., base_env=base_env, ...)")
    
    train_env = EdonHumanoidEnv(seed=42, profile="high_stress")
    train_obs = train_env.reset()
    print(f"\nTraining env type: {type(train_env).__name__}")
    print(f"Training env base type: {type(train_env.env).__name__}")
    print(f"Training obs keys: {list(train_obs.keys()) if isinstance(train_obs, dict) else 'N/A'}")
    
    # Evaluation environment (as used in run_eval.py)
    print("\n[2] Evaluation Environment Setup:")
    print("-"*80)
    print("Code: base_env = make_humanoid_env(seed=args.seed, profile=args.profile)")
    print("      env = EdonHumanoidEnvV8(..., base_env=base_env, ...)")
    
    eval_base_env = make_humanoid_env(seed=42, profile="high_stress")
    eval_obs = eval_base_env.reset()
    print(f"\nEval base env type: {type(eval_base_env).__name__}")
    print(f"Eval obs keys: {list(eval_obs.keys()) if isinstance(eval_obs, dict) else 'N/A'}")
    
    # Check if they're the same
    print("\n[3] Comparison:")
    print("-"*80)
    
    train_base_type = type(train_env.env).__name__
    eval_base_type = type(eval_base_env).__name__
    
    if train_base_type == eval_base_type:
        print(f"[OK] Base environment types match: {train_base_type}")
    else:
        print(f"[WARNING] Base environment types differ:")
        print(f"  Training: {train_base_type}")
        print(f"  Evaluation: {eval_base_type}")
    
    # Check observation structure
    if isinstance(train_obs, dict) and isinstance(eval_obs, dict):
        train_keys = set(train_obs.keys())
        eval_keys = set(eval_obs.keys())
        
        if train_keys == eval_keys:
            print(f"[OK] Observation keys match: {sorted(train_keys)}")
        else:
            print(f"[WARNING] Observation keys differ:")
            print(f"  Training only: {sorted(train_keys - eval_keys)}")
            print(f"  Evaluation only: {sorted(eval_keys - train_keys)}")
            print(f"  Common: {sorted(train_keys & eval_keys)}")
    
    # Check baseline controller output
    print("\n[4] Baseline Controller Check:")
    print("-"*80)
    
    train_baseline = baseline_controller(train_obs, edon_state=None)
    eval_baseline = baseline_controller(eval_obs, edon_state=None)
    
    train_baseline = np.array(train_baseline)
    eval_baseline = np.array(eval_baseline)
    
    if train_baseline.shape == eval_baseline.shape:
        print(f"[OK] Baseline action shapes match: {train_baseline.shape}")
        diff = np.abs(train_baseline - eval_baseline).max()
        if diff < 1e-6:
            print(f"[OK] Baseline actions are identical (max diff: {diff})")
        else:
            print(f"[WARNING] Baseline actions differ (max diff: {diff})")
    else:
        print(f"[WARNING] Baseline action shapes differ:")
        print(f"  Training: {train_baseline.shape}")
        print(f"  Evaluation: {eval_baseline.shape}")
    
    # Check stress profile application
    print("\n[5] Stress Profile Check:")
    print("-"*80)
    
    if hasattr(train_env.env, 'stress_profile'):
        train_profile = train_env.env.stress_profile
        print(f"Training stress profile: {train_profile}")
    else:
        print("Training env has no stress_profile attribute")
    
    if hasattr(eval_base_env, 'stress_profile'):
        eval_profile = eval_base_env.stress_profile
        print(f"Evaluation stress profile: {eval_profile}")
    else:
        print("Evaluation env has no stress_profile attribute")
    
    # Summary
    print("\n" + "="*80)
    print("Summary:")
    print("-"*80)
    
    if train_base_type == eval_base_type:
        print("[OK] Environments appear to be aligned")
    else:
        print("[WARNING] Environments may not be aligned - this could cause training/eval mismatch")
        print("\nRecommendation: Ensure both use the same base environment type")
    
    print("="*80)


if __name__ == "__main__":
    check_env_alignment()

