"""
EDON v8 Leverage Test (Direct - uses run_eval.py)

Tests whether EDON v8 can move interventions by running evaluations
with different gain multipliers.
"""

import sys
import subprocess
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.compare_v8_vs_baseline import load_results, extract_metrics


def run_eval_with_gain_multiplier(gain_multiplier, inverted, output_file, episodes=30, seed=42):
    """Run evaluation by temporarily modifying env code."""
    # We'll modify the environment to use gain_multiplier
    # For now, let's just run normal eval and manually compute what gain_multiplier would do
    # Actually, we already modified env/edon_humanoid_env_v8.py to accept gain_multiplier
    # So we need to modify run_eval.py to pass it, or create a wrapper
    
    # Simplest: create a small script that uses the env directly with gain_multiplier
    script_content = f"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_eval import make_humanoid_env, baseline_controller
from training.edon_v8_policy import EdonV8StrategyPolicy, pack_observation_v8
from training.fail_risk_model import FailRiskModel
from env.edon_humanoid_env_v8 import EdonHumanoidEnvV8
from evaluation.humanoid_runner import HumanoidRunner
from evaluation.metrics import EpisodeMetrics, aggregate_run_metrics
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
controller = baseline_controller
runner = HumanoidRunner(env=env, controller=controller)

episode_metrics_list = []
for ep_id in range({episodes}):
    obs = env.reset()
    episode_data = []
    done = False
    step = 0
    interventions = 0
    
    while not done and step < 1000:
        obs, reward, done, info = env.step(edon_core_state=None)
        # Check multiple intervention signals
        if info.get("intervention", False) or info.get("fallen", False) or info.get("freeze", False):
            interventions += 1
        episode_data.append({{"obs": obs, "info": info}})
        step += 1
        if done:
            break
    
    # Compute stability
    stability_sum = 0.0
    stability_count = 0
    for step_data in episode_data:
        obs = step_data["obs"]
        if isinstance(obs, dict):
            roll = abs(obs.get("roll", 0.0))
            pitch = abs(obs.get("pitch", 0.0))
            stability_sum += roll + pitch
            stability_count += 1
    
    stability = stability_sum / stability_count if stability_count > 0 else 0.0
    
    # Create EpisodeMetrics
    metrics = EpisodeMetrics(
        episode_id=ep_id,
        interventions=interventions,
        freeze_events=0,
        stability_score=stability,
        episode_length=step,
        success=False
    )
    episode_metrics_list.append(metrics)

# Aggregate
run_metrics = aggregate_run_metrics(episode_metrics_list, mode="edon")

# Save
results = {{
    "summary": {{
        "interventions_per_episode": run_metrics.interventions_per_episode,
        "stability": run_metrics.stability_avg,
        "episode_length": run_metrics.avg_episode_length
    }},
    "episodes": [m.to_dict() for m in episode_metrics_list]
}}

with open("{output_file}", "w") as f:
    json.dump(results, f, indent=2)

print(f"Results: Interventions/ep={{run_metrics.interventions_per_episode:.2f}}, Stability={{run_metrics.stability_avg:.4f}}")
"""
    
    test_file = Path("scripts/temp_leverage_test.py")
    test_file.write_text(script_content)
    
    result = subprocess.run(
        ["python", str(test_file)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    
    test_file.unlink()
    
    if result.returncode == 0:
        if Path(output_file).exists():
            return load_results(output_file)
    else:
        print(f"Error: {result.stderr}")
    
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
    cmd = [
        "python", "run_eval.py",
        "--mode", "baseline",
        "--profile", "high_stress",
        "--episodes", "30",
        "--seed", "42",
        "--output", "results/leverage_test_baseline.json",
        "--edon-score"
    ]
    subprocess.run(cmd, capture_output=True)
    baseline_results = load_results("results/leverage_test_baseline.json")
    if baseline_results:
        # Extract from run_metrics directly
        run_metrics = baseline_results.get("run_metrics", {})
        baseline_metrics = {
            "interventions_per_episode": run_metrics.get("interventions_per_episode", 0),
            "stability": run_metrics.get("stability_avg", 0)
        }
        results["baseline"] = baseline_metrics
        print(f"  Interventions/ep: {baseline_metrics.get('interventions_per_episode', 'N/A')}")
        print(f"  Stability: {baseline_metrics.get('stability', 'N/A')}")
    
    # Test 2-4: EDON with different gains
    configs = [
        ("edon_normal", 1.0, False),
        ("edon_high", 2.0, False),
        ("edon_inverted", -1.0, True)
    ]
    
    for config_name, gain, inverted in configs:
        print(f"\n[{configs.index((config_name, gain, inverted)) + 2}/4] EDON {config_name.replace('edon_', '').title()} (gain_multiplier={gain})")
        print("-"*80)
        output_file = f"results/leverage_test_{config_name}.json"
        test_results = run_eval_with_gain_multiplier(gain, inverted, output_file, episodes=30, seed=42)
        if test_results:
            # Extract metrics directly from summary
            summary = test_results.get("summary", {})
            metrics = {
                "interventions_per_episode": summary.get("interventions_per_episode", 0),
                "stability": summary.get("stability", 0)
            }
            results[config_name] = metrics
            print(f"  Interventions/ep: {metrics.get('interventions_per_episode', 'N/A')}")
            print(f"  Stability: {metrics.get('stability', 'N/A')}")
    
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

