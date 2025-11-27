"""Evaluate the new v8 memory+features model"""
import sys
from pathlib import Path
import torch
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent))

from run_eval import make_humanoid_env, baseline_controller
from training.edon_v8_policy import EdonV8StrategyPolicy
from training.fail_risk_model import FailRiskModel
from env.edon_humanoid_env_v8 import EdonHumanoidEnvV8
from evaluation.humanoid_runner import HumanoidRunner
from evaluation.metrics import aggregate_run_metrics
from metrics.edon_v8_metrics import compute_episode_score_v8, compute_episode_metrics_v8

def main():
    print("=" * 70)
    print("EDON v8 Memory+Features Model Evaluation")
    print("=" * 70)
    
    episodes = 30
    seed = 42  # Different from training (seed=0) to test generalization
    profile = "high_stress"
    
    # Create base environment
    print("\n[EVAL] Creating base environment...")
    base_env = make_humanoid_env(seed=seed, profile=profile)
    
    # Load fail-risk model
    print("[EVAL] Loading fail-risk model...")
    fail_risk_model = None
    fail_risk_path = Path("models/edon_fail_risk_v1_fixed_v2.pt")
    if fail_risk_path.exists():
        checkpoint = torch.load(fail_risk_path, map_location="cpu", weights_only=False)
        input_size = checkpoint.get("input_size", 15)
        fail_risk_model = FailRiskModel(input_size=input_size)
        fail_risk_model.load_state_dict(checkpoint["model_state_dict"])
        fail_risk_model.eval()
        print(f"  ✓ Loaded from {fail_risk_path}")
    else:
        print(f"  ⚠ Fail-risk model not found, using fail_risk=0.0")
    
    # Load strategy policy
    print("[EVAL] Loading v8 strategy policy...")
    strategy_path = Path("models/edon_v8_strategy_memory_features.pt")
    if not strategy_path.exists():
        print(f"  ✗ ERROR: Model not found at {strategy_path}")
        return
    
    checkpoint = torch.load(strategy_path, map_location="cpu", weights_only=False)
    input_size = checkpoint.get("input_size")
    
    if input_size is None:
        print("  ⚠ input_size not in checkpoint, inferring from env...")
        from training.edon_v8_policy import pack_stacked_observation_v8
        test_obs = base_env.reset()
        test_baseline = baseline_controller(test_obs, edon_state=None)
        test_input = pack_stacked_observation_v8(
            obs=test_obs,
            baseline_action=np.array(test_baseline),
            fail_risk=0.0,
            instability_score=0.0,
            phase="stable",
            obs_history=None,
            near_fail_history=None,
            obs_vec_history=None,
            stack_size=8
        )
        input_size = len(test_input)
        print(f"  ✓ Inferred input_size: {input_size}")
    else:
        print(f"  ✓ input_size from checkpoint: {input_size}")
    
    policy = EdonV8StrategyPolicy(input_size=input_size)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    print(f"  ✓ Policy loaded successfully")
    
    # Create v8 environment
    print("[EVAL] Creating v8 environment...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = EdonHumanoidEnvV8(
        strategy_policy=policy,
        fail_risk_model=fail_risk_model,
        base_env=base_env,
        seed=seed,
        profile=profile,
        device=device
    )
    print(f"  ✓ Environment created (device: {device})")
    
    # Run evaluation
    print(f"\n[EVAL] Running {episodes} episodes...")
    print("=" * 70)
    
    episodes_data = []
    for ep_id in range(episodes):
        obs = env.reset()
        episode_data = []
        done = False
        step = 0
        
        while not done and step < 1000:
            obs, reward, done, info = env.step(edon_core_state=None)
            episode_data.append({"obs": obs, "info": info, "done": done})
            step += 1
            if done:
                break
        
        # Compute v8 metrics
        v8_metrics = compute_episode_metrics_v8(episode_data)
        episodes_data.append(v8_metrics)
        
        if (ep_id + 1) % 10 == 0:
            print(f"[EVAL] Completed {ep_id + 1}/{episodes} episodes...")
    
    # Aggregate results
    print("\n[EVAL] Computing aggregate metrics...")
    
    # Compute summary from dictionaries (v8 metrics are dicts, not EpisodeMetrics objects)
    interventions = [ep.get("interventions", 0) for ep in episodes_data]
    freeze_events = [ep.get("freeze_events", 0) for ep in episodes_data]
    stability_scores = [ep.get("stability_score", 0.0) for ep in episodes_data]
    episode_lengths = [ep.get("episode_length", 0) for ep in episodes_data]
    successes = [1 if ep.get("success", False) else 0 for ep in episodes_data]
    
    summary = {
        "interventions_total": sum(interventions),
        "interventions_per_episode": np.mean(interventions) if interventions else 0.0,
        "freeze_events_total": sum(freeze_events),
        "freeze_events_per_episode": np.mean(freeze_events) if freeze_events else 0.0,
        "stability_avg": np.mean(stability_scores) if stability_scores else 0.0,
        "stability_std": np.std(stability_scores) if stability_scores else 0.0,
        "avg_episode_length": np.mean(episode_lengths) if episode_lengths else 0.0,
        "success_rate": np.mean(successes) if successes else 0.0,
    }
    
    # Compute v8 score
    v8_score = compute_episode_score_v8(summary)
    summary["edon_score_v8"] = v8_score
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nEpisodes: {episodes}")
    print(f"Profile: {profile}")
    print(f"Seed: {seed}")
    print(f"\nMetrics:")
    print(f"  Interventions/episode: {summary['interventions_per_episode']:.2f}")
    print(f"  Stability (avg): {summary.get('stability_avg', 0.0):.4f}")
    print(f"  Episode length (avg): {summary.get('avg_episode_length', 0.0):.1f}")
    print(f"  EDON v8 Score: {v8_score:.2f}")
    
    # Save results
    output_path = Path("results/edon_v8_memory_features.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            "mode": "edon",
            "profile": profile,
            "episodes": episodes,
            "seed": seed,
            "model": "edon_v8_strategy_memory_features",
            "run_metrics": summary,
            "episodes_data": episodes_data
        }, f, indent=2)
    
    print(f"\n[EVAL] Results saved to {output_path}")
    print("=" * 70)
    
    # Compare with baseline (if available)
    baseline_path = Path("results/baseline_high_stress.json")
    if not baseline_path.exists():
        baseline_path = Path("results/baseline_high_v42.json")
    
    if baseline_path.exists():
        print("\n" + "=" * 70)
        print("COMPARISON WITH BASELINE")
        print("=" * 70)
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
        
        baseline_int = baseline.get("interventions_per_episode", baseline.get("summary", {}).get("interventions_per_episode", 0))
        baseline_stab = baseline.get("stability_avg", baseline.get("summary", {}).get("stability", 0))
        
        edon_int = summary["interventions_per_episode"]
        edon_stab = summary.get("stability_avg", 0.0)
        
        delta_int = 100 * (baseline_int - edon_int) / baseline_int if baseline_int > 0 else 0
        delta_stab = 100 * (baseline_stab - edon_stab) / baseline_stab if baseline_stab > 0 else 0
        
        print(f"\nBaseline:")
        print(f"  Interventions/ep: {baseline_int:.2f}")
        print(f"  Stability: {baseline_stab:.4f}")
        print(f"\nEDON v8 Memory+Features:")
        print(f"  Interventions/ep: {edon_int:.2f}")
        print(f"  Stability: {edon_stab:.4f}")
        print(f"\nDelta:")
        print(f"  ΔInterventions: {delta_int:+.2f}% {'✅' if delta_int >= 10 else '❌'} (target: ≥10%)")
        print(f"  ΔStability: {delta_stab:+.2f}% {'✅' if abs(delta_stab) <= 5 else '❌'} (target: ±5%)")
        
        if delta_int >= 10 and abs(delta_stab) <= 5:
            print(f"\n✅ GOAL ACHIEVED: ≥10% intervention reduction with stable stability!")
        else:
            print(f"\n⚠️  Goal not fully met yet.")
        print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[EVAL] ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise

