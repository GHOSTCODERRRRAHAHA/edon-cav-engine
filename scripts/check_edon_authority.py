"""
Check EDON Control Authority

Diagnostic script to check if EDON has enough control authority
to meaningfully affect interventions.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_eval import make_humanoid_env, baseline_controller
from training.edon_v8_policy import EdonV8StrategyPolicy, pack_observation_v8
from training.fail_risk_model import FailRiskModel
from env.edon_humanoid_env_v8 import EdonHumanoidEnvV8
import torch


def check_edon_authority():
    """Check EDON's control authority vs baseline."""
    print("="*80)
    print("EDON Control Authority Diagnostic")
    print("="*80)
    
    # Create environment
    base_env = make_humanoid_env(seed=42, profile="high_stress")
    
    # Load fail-risk model
    fail_risk_model = None
    fail_risk_path = Path("models/edon_fail_risk_v1_fixed_v2.pt")
    if fail_risk_path.exists():
        checkpoint = torch.load(fail_risk_path, map_location="cpu", weights_only=False)
        input_size = checkpoint.get("input_size", 15)
        fail_risk_model = FailRiskModel(input_size=input_size)
        fail_risk_model.load_state_dict(checkpoint["model_state_dict"])
        fail_risk_model.eval()
    
    # Load strategy policy
    strategy_path = Path("models/edon_v8_strategy_intervention_first.pt")
    if not strategy_path.exists():
        print(f"ERROR: Strategy model not found: {strategy_path}")
        return
    
    checkpoint = torch.load(strategy_path, map_location="cpu", weights_only=False)
    input_size = checkpoint.get("input_size", 25)
    policy = EdonV8StrategyPolicy(input_size=input_size)
    
    if "policy_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["policy_state_dict"])
    else:
        policy.load_state_dict(checkpoint)
    policy.eval()
    
    # Test with different EDON gains
    print("\n[1] Testing EDON action magnitude vs baseline")
    print("-"*80)
    
    obs = base_env.reset()
    baseline_action = baseline_controller(obs, edon_state=None)
    baseline_action = np.array(baseline_action)
    
    # Get EDON strategy output
    obs_vec = pack_observation_v8(
        obs=obs,
        baseline_action=baseline_action,
        fail_risk=0.0,
        instability_score=0.0,
        phase="stable"
    )
    
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(obs_vec).unsqueeze(0)
        strategy_id, modulations, _ = policy.sample_action(obs_tensor, deterministic=True)
    
    # Apply modulations
    edon_action = baseline_action.copy()
    if modulations:
        gain_scale = modulations.get("gain_scale", 1.0)
        edon_action = edon_action * gain_scale
        
        lateral_compliance = modulations.get("lateral_compliance", 1.0)
        if len(edon_action) >= 4:
            edon_action[:4] = edon_action[:4] * lateral_compliance
        
        step_height_bias = modulations.get("step_height_bias", 0.0)
        if len(edon_action) >= 8:
            edon_action[4:8] = edon_action[4:8] + step_height_bias * 0.1
    
    edon_action = np.clip(edon_action, -1.0, 1.0)
    
    # Compute differences
    action_delta = edon_action - baseline_action
    baseline_norm = np.linalg.norm(baseline_action)
    delta_norm = np.linalg.norm(action_delta)
    delta_percent = (delta_norm / baseline_norm * 100) if baseline_norm > 0 else 0.0
    
    print(f"Baseline action norm: {baseline_norm:.4f}")
    print(f"EDON action norm: {np.linalg.norm(edon_action):.4f}")
    print(f"Action delta norm: {delta_norm:.4f}")
    print(f"Delta as % of baseline: {delta_percent:.2f}%")
    print(f"Strategy: {EdonV8StrategyPolicy.STRATEGIES[strategy_id]}")
    print(f"Modulations: gain_scale={modulations.get('gain_scale', 1.0):.3f}, "
          f"lateral={modulations.get('lateral_compliance', 1.0):.3f}, "
          f"step_height={modulations.get('step_height_bias', 0.0):.3f}")
    
    # Test with 2x gain
    print("\n[2] Testing with 2x EDON gain (temporary diagnostic)")
    print("-"*80)
    
    edon_action_2x = baseline_action.copy()
    if modulations:
        gain_scale_2x = modulations.get("gain_scale", 1.0) * 2.0  # 2x gain
        edon_action_2x = edon_action_2x * gain_scale_2x
        
        lateral_compliance = modulations.get("lateral_compliance", 1.0)
        if len(edon_action_2x) >= 4:
            edon_action_2x[:4] = edon_action_2x[:4] * lateral_compliance
        
        step_height_bias = modulations.get("step_height_bias", 0.0)
        if len(edon_action_2x) >= 8:
            edon_action_2x[4:8] = edon_action_2x[4:8] + step_height_bias * 0.1
    
    edon_action_2x = np.clip(edon_action_2x, -1.0, 1.0)
    delta_2x = edon_action_2x - baseline_action
    delta_2x_norm = np.linalg.norm(delta_2x)
    delta_2x_percent = (delta_2x_norm / baseline_norm * 100) if baseline_norm > 0 else 0.0
    
    print(f"2x gain action delta norm: {delta_2x_norm:.4f}")
    print(f"2x gain delta as % of baseline: {delta_2x_percent:.2f}%")
    
    # Run short diagnostic episodes
    print("\n[3] Running diagnostic episodes (10 steps each)")
    print("-"*80)
    
    # Baseline only
    baseline_interventions = []
    obs = base_env.reset()
    for step in range(10):
        action = baseline_controller(obs, edon_state=None)
        obs, _, done, info = base_env.step(np.array(action))
        if info.get("intervention", False):
            baseline_interventions.append(step)
        if done:
            break
    
    # EDON normal
    edon_interventions = []
    env_v8 = EdonHumanoidEnvV8(
        strategy_policy=policy,
        fail_risk_model=fail_risk_model,
        base_env=base_env,
        seed=42,
        profile="high_stress",
        w_intervention=10.0,
        w_stability=1.0,
        w_torque=0.1
    )
    obs = env_v8.reset()
    for step in range(10):
        obs, _, done, info = env_v8.step(edon_core_state=None)
        if info.get("intervention", False):
            edon_interventions.append(step)
        if done:
            break
    
    print(f"Baseline interventions in 10 steps: {len(baseline_interventions)}")
    print(f"EDON interventions in 10 steps: {len(edon_interventions)}")
    
    # Assessment
    print("\n" + "="*80)
    print("Assessment:")
    print("-"*80)
    
    if delta_percent < 1.0:
        print("[WARNING] EDON action delta is <1% of baseline - very weak control authority")
        print("  Recommendation: Increase EDON gain or modulation range")
    elif delta_percent < 5.0:
        print("[CAUTION] EDON action delta is 1-5% of baseline - moderate control authority")
        print("  May need to increase modulation range or gain")
    else:
        print(f"[OK] EDON action delta is {delta_percent:.1f}% of baseline - reasonable control authority")
    
    if delta_2x_percent < 5.0:
        print("[WARNING] Even with 2x gain, delta is <5% - EDON may be fundamentally limited")
    else:
        print(f"[OK] With 2x gain, delta is {delta_2x_percent:.1f}% - EDON can scale up")
    
    print("="*80)


if __name__ == "__main__":
    check_edon_authority()

