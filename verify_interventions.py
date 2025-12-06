"""Verify that interventions are being detected correctly"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from run_eval import make_humanoid_env, baseline_controller
from training.edon_v8_policy import EdonV8StrategyPolicy
from training.fail_risk_model import FailRiskModel
from env.edon_humanoid_env_v8 import EdonHumanoidEnvV8
import torch

print("="*70)
print("VERIFYING INTERVENTION DETECTION")
print("="*70)

# Test 1: Check baseline environment directly
print("\n[TEST 1] Checking baseline environment intervention detection...")
base_env = make_humanoid_env(seed=42, profile="high_stress")
obs = base_env.reset()

intervention_count = 0
fallen_count = 0
step_count = 0

for i in range(100):
    # Use a bad action that should cause interventions
    bad_action = np.ones(10) * 2.0  # Large action that should cause issues
    obs, reward, done, info = base_env.step(bad_action)
    
    if isinstance(info, dict):
        if info.get("intervention", False):
            intervention_count += 1
            print(f"  Step {i}: intervention=True")
        if info.get("fallen", False):
            fallen_count += 1
            print(f"  Step {i}: fallen=True")
    
    step_count += 1
    if done:
        obs = base_env.reset()

print(f"\n  Total steps: {step_count}")
print(f"  Interventions detected: {intervention_count}")
print(f"  Fallen events detected: {fallen_count}")

# Test 2: Check v8 environment
print("\n[TEST 2] Checking v8 environment intervention detection...")

# Load models
fail_risk_model = None
fail_risk_path = Path("models/edon_fail_risk_v1_fixed_v2.pt")
if fail_risk_path.exists():
    checkpoint = torch.load(fail_risk_path, map_location="cpu", weights_only=False)
    input_size = checkpoint.get("input_size", 15)
    fail_risk_model = FailRiskModel(input_size=input_size)
    fail_risk_model.load_state_dict(checkpoint["model_state_dict"])
    fail_risk_model.eval()

strategy_path = Path("models/edon_v8_strategy_memory_features.pt")
if strategy_path.exists():
    checkpoint = torch.load(strategy_path, map_location="cpu", weights_only=False)
    input_size = checkpoint.get("input_size", 248)
    policy = EdonV8StrategyPolicy(input_size=input_size)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    
    device = "cpu"
    v8_env = EdonHumanoidEnvV8(
        strategy_policy=policy,
        fail_risk_model=fail_risk_model,
        base_env=make_humanoid_env(seed=42, profile="high_stress"),
        seed=42,
        profile="high_stress",
        device=device
    )
    
    obs = v8_env.reset()
    v8_intervention_count = 0
    v8_fallen_count = 0
    v8_step_count = 0
    
    for i in range(100):
        obs, reward, done, info = v8_env.step(edon_core_state=None)
        
        if isinstance(info, dict):
            if info.get("intervention", False):
                v8_intervention_count += 1
                print(f"  Step {i}: intervention=True, roll={obs.get('roll', 0):.3f}, pitch={obs.get('pitch', 0):.3f}")
            if info.get("fallen", False):
                v8_fallen_count += 1
                print(f"  Step {i}: fallen=True, roll={obs.get('roll', 0):.3f}, pitch={obs.get('pitch', 0):.3f}")
        
        v8_step_count += 1
        if done:
            obs = v8_env.reset()
    
    print(f"\n  Total steps: {v8_step_count}")
    print(f"  Interventions detected: {v8_intervention_count}")
    print(f"  Fallen events detected: {v8_fallen_count}")
    
    # Test 3: Check what the metrics function sees
    print("\n[TEST 3] Checking what compute_episode_metrics_v8 sees...")
    from metrics.edon_v8_metrics import compute_episode_metrics_v8
    
    # Collect one episode
    obs = v8_env.reset()
    episode_data = []
    done = False
    step = 0
    
    while not done and step < 200:
        obs, reward, done, info = v8_env.step(edon_core_state=None)
        episode_data.append({"obs": obs, "info": info, "done": done})
        step += 1
        if done:
            break
    
    metrics = compute_episode_metrics_v8(episode_data)
    print(f"\n  Episode metrics:")
    print(f"    interventions: {metrics.get('interventions', 0)}")
    print(f"    episode_length: {metrics.get('episode_length', 0)}")
    
    # Check info dicts
    intervention_steps = []
    for i, step_data in enumerate(episode_data):
        info = step_data.get("info", {})
        if isinstance(info, dict):
            if info.get("intervention", False) or info.get("fallen", False):
                intervention_steps.append(i)
    
    print(f"    Steps with intervention/fallen flags: {len(intervention_steps)}")
    if intervention_steps:
        print(f"    First 5 intervention steps: {intervention_steps[:5]}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Baseline env: {intervention_count} interventions, {fallen_count} fallen in {step_count} steps")
    print(f"V8 env: {v8_intervention_count} interventions, {v8_fallen_count} fallen in {v8_step_count} steps")
    print(f"Metrics function: {metrics.get('interventions', 0)} interventions detected")
    
    if v8_intervention_count == 0 and v8_fallen_count == 0 and metrics.get('interventions', 0) == 0:
        print("\n⚠️  WARNING: No interventions detected! This could mean:")
        print("   1. The policy is actually preventing all interventions (good!)")
        print("   2. Intervention detection is broken (bad!)")
        print("   3. The test was too short or actions weren't bad enough")
    else:
        print("\n✅ Interventions are being detected correctly")
    
    print("="*70)

else:
    print("  ✗ Model not found, skipping v8 test")

