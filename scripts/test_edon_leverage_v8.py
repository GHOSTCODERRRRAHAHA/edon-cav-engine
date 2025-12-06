"""
EDON v8 Leverage Test (Direct)

Tests whether EDON v8 can move interventions by directly modifying
the environment's gain application in code.
"""

import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_eval import make_humanoid_env, baseline_controller
from training.edon_v8_policy import EdonV8StrategyPolicy, pack_observation_v8
from training.fail_risk_model import FailRiskModel
from env.edon_humanoid_env_v8 import EdonHumanoidEnvV8
from evaluation.humanoid_runner import HumanoidRunner
from evaluation.metrics import aggregate_run_metrics
from training.compare_v8_vs_baseline import extract_metrics


def run_v8_evaluation_with_gain(gain_multiplier: float = 1.0, inverted: bool = False,
                                 episodes: int = 30, seed: int = 42) -> Dict[str, Any]:
    """Run v8 evaluation with modified gain."""
    print(f"\nRunning v8 evaluation: gain_multiplier={gain_multiplier}, inverted={inverted}")
    
    # Create base environment
    base_env = make_humanoid_env(seed=seed, profile="high_stress")
    
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
        return None
    
    checkpoint = torch.load(strategy_path, map_location="cpu", weights_only=False)
    input_size = checkpoint.get("input_size", 25)
    policy = EdonV8StrategyPolicy(input_size=input_size)
    
    if "policy_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["policy_state_dict"])
    else:
        policy.load_state_dict(checkpoint)
    policy.eval()
    
    # Create v8 environment
    env = EdonHumanoidEnvV8(
        strategy_policy=policy,
        fail_risk_model=fail_risk_model,
        base_env=base_env,
        seed=seed,
        profile="high_stress",
        device="cpu"
    )
    
    # Monkey-patch the step method to apply gain multiplier
    original_step = env.step
    
    def modified_step(edon_core_state=None):
        """Modified step that applies gain multiplier to modulations."""
        obs, reward, done, info = original_step(edon_core_state)
        
        # Apply gain multiplier to the action that was just applied
        # We need to modify the action before it's applied
        # Actually, we need to modify it in the step method itself
        return obs, reward, done, info
    
    # Better approach: modify the action computation in step
    original_compute_action = None
    if hasattr(env, 'compute_action'):
        original_compute_action = env.compute_action
    
    # Actually, let's modify the modulations directly in the step method
    # We'll patch the part where modulations are applied
    def patched_step(self, action=None, edon_core_state=None):
        """Patched step with gain multiplier."""
        # Get current observation
        if self.prev_obs is None:
            obs = self.env.reset()
        else:
            obs = self.prev_obs
        
        # Get baseline action
        baseline_action = baseline_controller(obs, edon_state=None)
        baseline_action = np.array(baseline_action)
        
        # Compute fail-risk
        features = None
        if edon_core_state:
            features = {
                "instability_score": edon_core_state.get("instability_score", 0.0),
                "risk_ema": edon_core_state.get("risk_ema", 0.0),
                "phase": edon_core_state.get("phase", "stable")
            }
        
        fail_risk = self.compute_fail_risk(obs, features)
        self.current_fail_risk = fail_risk
        
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
            strategy_id, modulations, _ = self.strategy_policy.sample_action(obs_tensor, deterministic=True)
        
        self.current_strategy_id = strategy_id
        self.current_modulations = modulations
        
        # Apply modulations with gain multiplier
        final_action = baseline_action.copy()
        if modulations:
            gain_scale = modulations.get("gain_scale", 1.0)
            # Apply gain multiplier
            gain_scale = gain_scale * gain_multiplier
            if inverted:
                gain_scale = -gain_scale  # Invert
            
            final_action = final_action * gain_scale
            
            lateral_compliance = modulations.get("lateral_compliance", 1.0)
            if len(final_action) >= 4:
                final_action[:4] = final_action[:4] * lateral_compliance
            
            step_height_bias = modulations.get("step_height_bias", 0.0)
            if len(final_action) >= 8:
                final_action[4:8] = final_action[4:8] + step_height_bias * 0.1
        
        # Clip to valid range
        final_action = np.clip(final_action, -1.0, 1.0)
        
        # Step base environment
        next_obs, reward, done, info = self.env.step(final_action)
        
        # Add v8-specific info
        info["fail_risk"] = fail_risk
        info["strategy_id"] = strategy_id
        if modulations:
            info["gain_scale"] = modulations.get("gain_scale", 1.0)
            info["lateral_compliance"] = modulations.get("lateral_compliance", 1.0)
        
        # Compute EDON reward
        from training.edon_score import step_reward
        edon_reward = step_reward(
            prev_state=self.prev_obs,
            next_state=next_obs,
            info=info,
            w_intervention=20.0,
            w_stability=1.0,
            w_torque=0.1
        )
        
        self.prev_obs = next_obs
        return next_obs, edon_reward, done, info
    
    # Replace step method
    import types
    env.step = types.MethodType(patched_step, env)
    
    # Run evaluation
    runner = HumanoidRunner(env=env)
    episodes_data = []
    
    for ep in range(episodes):
        obs = env.reset()
        episode_data = []
        done = False
        step = 0
        
        while not done and step < 1000:
            obs, reward, done, info = env.step(edon_core_state=None)
            episode_data.append({
                "obs": obs,
                "reward": reward,
                "done": done,
                "info": info
            })
            step += 1
        
        episodes_data.append(episode_data)
    
    # Aggregate metrics
    run_metrics = aggregate_run_metrics(episodes_data)
    
    # Convert to expected format
    results = {
        "episodes": [
            {
                "interventions": sum(1 for step in ep if step["info"].get("intervention", False)),
                "stability": np.mean([abs(step["obs"].get("roll", 0)) + abs(step["obs"].get("pitch", 0)) 
                                     for step in ep if isinstance(step["obs"], dict)]),
                "length": len(ep)
            }
            for ep in episodes_data
        ],
        "summary": {
            "interventions_per_episode": run_metrics.interventions_per_episode,
            "stability": run_metrics.stability,
            "episode_length": run_metrics.episode_length
        }
    }
    
    return results


def test_edon_leverage_v8():
    """Run leverage test with v8."""
    print("="*80)
    print("EDON v8 Leverage Test")
    print("="*80)
    
    results = {}
    
    # Test 1: Baseline (EDON OFF - use baseline controller directly)
    print("\n[1/4] Baseline Only (EDON OFF)")
    print("-"*80)
    # Run baseline evaluation
    cmd = [
        "python", "run_eval.py",
        "--mode", "baseline",
        "--profile", "high_stress",
        "--episodes", "30",
        "--seed", "42",
        "--output", "results/leverage_test_baseline.json",
        "--edon-score"
    ]
    subprocess.run(cmd)
    baseline_results = load_results("results/leverage_test_baseline.json")
    if baseline_results:
        results["baseline"] = extract_metrics(baseline_results)
    
    # Test 2: EDON normal (gain_multiplier=1.0)
    print("\n[2/4] EDON Normal (gain_multiplier=1.0)")
    print("-"*80)
    normal_results = run_v8_evaluation_with_gain(gain_multiplier=1.0, episodes=30, seed=42)
    if normal_results:
        results["edon_normal"] = extract_metrics(normal_results)
    
    # Test 3: EDON high (gain_multiplier=2.0)
    print("\n[3/4] EDON High Gain (gain_multiplier=2.0)")
    print("-"*80)
    high_results = run_v8_evaluation_with_gain(gain_multiplier=2.0, episodes=30, seed=42)
    if high_results:
        results["edon_high"] = extract_metrics(high_results)
    
    # Test 4: EDON inverted (gain_multiplier=-1.0)
    print("\n[4/4] EDON Inverted (gain_multiplier=-1.0)")
    print("-"*80)
    inverted_results = run_v8_evaluation_with_gain(gain_multiplier=-1.0, inverted=True, episodes=30, seed=42)
    if inverted_results:
        results["edon_inverted"] = extract_metrics(inverted_results)
    
    # Print summary
    print("\n" + "="*80)
    print("Leverage Test Summary")
    print("="*80)
    
    baseline_int = results.get("baseline", {}).get("interventions_per_episode", 0)
    
    print(f"\n{'Config':<20} {'Interventions/ep':<20} {'Stability':<15} {'Delta%':<15}")
    print("-"*80)
    
    for config_name, metrics in results.items():
        int_val = metrics.get("interventions_per_episode", 0)
        stab_val = metrics.get("stability", 0)
        
        if baseline_int > 0:
            delta_pct = ((int_val - baseline_int) / baseline_int) * 100
        else:
            delta_pct = 0.0
        
        print(f"{config_name:<20} {int_val:<20.2f} {stab_val:<15.4f} {delta_pct:>+6.1f}%")
    
    # Analysis
    print("\n" + "="*80)
    print("Analysis")
    print("="*80)
    
    if baseline_int > 0:
        normal_int = results.get("edon_normal", {}).get("interventions_per_episode", baseline_int)
        high_int = results.get("edon_high", {}).get("interventions_per_episode", baseline_int)
        inverted_int = results.get("edon_inverted", {}).get("interventions_per_episode", baseline_int)
        
        normal_delta = abs((normal_int - baseline_int) / baseline_int * 100)
        high_delta = abs((high_int - baseline_int) / baseline_int * 100)
        inverted_delta = abs((inverted_int - baseline_int) / baseline_int * 100)
        
        max_delta = max(normal_delta, high_delta, inverted_delta)
        
        if max_delta < 2.0:
            print("\n[CONCLUSION] EDON has WEAK leverage (<2% change)")
            print("  - Interventions barely change with EDON on/off/high/inverted")
            print("  - EDON's hook doesn't affect intervention triggers")
            print("\n[RECOMMENDATION] Change where EDON acts (move up a level)")
        elif max_delta >= 10.0:
            print(f"\n[CONCLUSION] EDON has STRONG leverage ({max_delta:.1f}% change)")
            print("  - Control hook is fine, policy needs better features/context")
            print("\n[RECOMMENDATION] Fix what EDON sees (add memory, early-warning features)")
        else:
            print(f"\n[CONCLUSION] EDON has MODERATE leverage ({max_delta:.1f}% change)")
    
    print("="*80)


def load_results(file_path: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    test_edon_leverage_v8()

