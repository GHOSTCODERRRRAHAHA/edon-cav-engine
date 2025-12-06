"""
EDON v8 Leverage Test (Simple)

Tests whether EDON v8 can move interventions by running evaluations
with different gain multipliers using run_eval.py infrastructure.
"""

import sys
import subprocess
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.compare_v8_vs_baseline import load_results, extract_metrics


def run_baseline(episodes=30, seed=42):
    """Run baseline evaluation."""
    output_file = "results/leverage_test_baseline.json"
    cmd = [
        "python", "run_eval.py",
        "--mode", "baseline",
        "--profile", "high_stress",
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--output", output_file,
        "--edon-score"
    ]
    subprocess.run(cmd, capture_output=True)
    return load_results(output_file)


def run_v8_with_gain_multiplier(gain_multiplier, inverted=False, episodes=30, seed=42):
    """Run v8 evaluation with modified gain multiplier."""
    # We need to modify the environment code temporarily
    # For now, let's create a wrapper script that patches the env
    
    # Actually, let's just modify run_eval.py to accept a gain_multiplier parameter
    # Or create a simple test that directly instantiates the env with gain_multiplier
    
    # For simplicity, let's use run_eval.py but we'll need to modify env/edon_humanoid_env_v8.py
    # to accept gain_multiplier parameter
    
    # Since we already modified the env, we can create a simple test script
    output_file = f"results/leverage_test_v8_gain{gain_multiplier}{'_inv' if inverted else ''}.json"
    
    # Create a simple test script
    test_script = f"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_eval import make_humanoid_env, baseline_controller
from training.edon_v8_policy import EdonV8StrategyPolicy, pack_observation_v8
from training.fail_risk_model import FailRiskModel
from env.edon_humanoid_env_v8 import EdonHumanoidEnvV8
from evaluation.humanoid_runner import HumanoidRunner
from evaluation.metrics import aggregate_run_metrics
import torch
import numpy as np
import json

# Create base environment
base_env = make_humanoid_env(seed={seed}, profile="high_stress")

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
checkpoint = torch.load(strategy_path, map_location="cpu", weights_only=False)
input_size = checkpoint.get("input_size", 25)
policy = EdonV8StrategyPolicy(input_size=input_size)
if "policy_state_dict" in checkpoint:
    policy.load_state_dict(checkpoint["policy_state_dict"])
else:
    policy.load_state_dict(checkpoint)
policy.eval()

# Create v8 environment with gain multiplier
env = EdonHumanoidEnvV8(
    strategy_policy=policy,
    fail_risk_model=fail_risk_model,
    base_env=base_env,
    seed={seed},
    profile="high_stress",
    device="cpu",
    gain_multiplier={gain_multiplier},
    inverted={str(inverted)}
)

# Run evaluation
from run_eval import baseline_controller
controller = baseline_controller
runner = HumanoidRunner(env=env, controller=controller)

episodes_data = []
for ep in range({episodes}):
    obs = env.reset()
    episode_data = []
    done = False
    step = 0
    while not done and step < 1000:
        obs, reward, done, info = env.step(edon_core_state=None)
        episode_data.append({{
            "obs": obs,
            "reward": reward,
            "done": done,
            "info": info
        }})
        step += 1
    episodes_data.append(episode_data)

# Aggregate metrics
run_metrics = aggregate_run_metrics(episodes_data, mode="edon")

# Save results
results = {{
    "summary": {{
        "interventions_per_episode": run_metrics.interventions_per_episode,
        "stability": run_metrics.stability,
        "episode_length": run_metrics.episode_length
    }},
    "episodes": [
        {{
            "interventions": sum(1 for step in ep if step["info"].get("intervention", False)),
            "stability": np.mean([abs(step["obs"].get("roll", 0)) + abs(step["obs"].get("pitch", 0)) 
                                 for step in ep if isinstance(step["obs"], dict)]),
            "length": len(ep)
        }}
        for ep in episodes_data
    ]
}}

with open("{output_file}", "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {output_file}")
"""
    
    # Write and run test script
    test_file = Path("scripts/temp_leverage_test.py")
    test_file.write_text(test_script)
    
    result = subprocess.run(
        ["python", str(test_file)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    
    # Clean up
    test_file.unlink()
    
    if Path(output_file).exists():
        return load_results(output_file)
    return None


def test_edon_leverage():
    """Run leverage test."""
    print("="*80)
    print("EDON v8 Leverage Test")
    print("="*80)
    
    results = {}
    
    # Test 1: Baseline
    print("\n[1/4] Baseline Only (EDON OFF)")
    print("-"*80)
    baseline_results = run_baseline(episodes=30, seed=42)
    if baseline_results:
        results["baseline"] = extract_metrics(baseline_results)
        print(f"  Interventions/ep: {results['baseline'].get('interventions_per_episode', 'N/A')}")
        print(f"  Stability: {results['baseline'].get('stability', 'N/A')}")
    
    # Test 2: EDON normal (gain_multiplier=1.0)
    print("\n[2/4] EDON Normal (gain_multiplier=1.0)")
    print("-"*80)
    normal_results = run_v8_with_gain_multiplier(1.0, inverted=False, episodes=30, seed=42)
    if normal_results:
        results["edon_normal"] = extract_metrics(normal_results)
        print(f"  Interventions/ep: {results['edon_normal'].get('interventions_per_episode', 'N/A')}")
        print(f"  Stability: {results['edon_normal'].get('stability', 'N/A')}")
    
    # Test 3: EDON high (gain_multiplier=2.0)
    print("\n[3/4] EDON High Gain (gain_multiplier=2.0)")
    print("-"*80)
    high_results = run_v8_with_gain_multiplier(2.0, inverted=False, episodes=30, seed=42)
    if high_results:
        results["edon_high"] = extract_metrics(high_results)
        print(f"  Interventions/ep: {results['edon_high'].get('interventions_per_episode', 'N/A')}")
        print(f"  Stability: {results['edon_high'].get('stability', 'N/A')}")
    
    # Test 4: EDON inverted (gain_multiplier=-1.0)
    print("\n[4/4] EDON Inverted (gain_multiplier=-1.0)")
    print("-"*80)
    inverted_results = run_v8_with_gain_multiplier(-1.0, inverted=True, episodes=30, seed=42)
    if inverted_results:
        results["edon_inverted"] = extract_metrics(inverted_results)
        print(f"  Interventions/ep: {results['edon_inverted'].get('interventions_per_episode', 'N/A')}")
        print(f"  Stability: {results['edon_inverted'].get('stability', 'N/A')}")
    
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


if __name__ == "__main__":
    test_edon_leverage()

