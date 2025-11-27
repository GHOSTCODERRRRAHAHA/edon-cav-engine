"""Evaluate v8 model on multiple seeds to test generalization"""
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
from metrics.edon_v8_metrics import compute_episode_score_v8, compute_episode_metrics_v8

def evaluate_seed(seed, episodes=30):
    """Evaluate on a specific seed"""
    print(f"\n{'='*70}")
    print(f"Evaluating with seed={seed}")
    print(f"{'='*70}")
    
    profile = "high_stress"
    
    # Create base environment
    base_env = make_humanoid_env(seed=seed, profile=profile)
    
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
    strategy_path = Path("models/edon_v8_strategy_memory_features.pt")
    if not strategy_path.exists():
        print(f"  ✗ ERROR: Model not found at {strategy_path}")
        return None
    
    checkpoint = torch.load(strategy_path, map_location="cpu", weights_only=False)
    input_size = checkpoint.get("input_size", 248)
    
    policy = EdonV8StrategyPolicy(input_size=input_size)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    
    # Create v8 environment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = EdonHumanoidEnvV8(
        strategy_policy=policy,
        fail_risk_model=fail_risk_model,
        base_env=base_env,
        seed=seed,
        profile=profile,
        device=device
    )
    
    # Run evaluation
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
            print(f"  Completed {ep_id + 1}/{episodes} episodes...")
    
    # Compute summary
    interventions = [ep.get("interventions", 0) for ep in episodes_data]
    freeze_events = [ep.get("freeze_events", 0) for ep in episodes_data]
    stability_scores = [ep.get("stability_score", 0.0) for ep in episodes_data]
    episode_lengths = [ep.get("episode_length", 0) for ep in episodes_data]
    successes = [1 if ep.get("success", False) else 0 for ep in episodes_data]
    
    summary = {
        "seed": seed,
        "interventions_total": sum(interventions),
        "interventions_per_episode": np.mean(interventions) if interventions else 0.0,
        "freeze_events_total": sum(freeze_events),
        "freeze_events_per_episode": np.mean(freeze_events) if freeze_events else 0.0,
        "stability_avg": np.mean(stability_scores) if stability_scores else 0.0,
        "stability_std": np.std(stability_scores) if stability_scores else 0.0,
        "avg_episode_length": np.mean(episode_lengths) if episode_lengths else 0.0,
        "success_rate": np.mean(successes) if successes else 0.0,
    }
    
    summary["edon_score_v8"] = compute_episode_score_v8(summary)
    
    return summary

def main():
    print("="*70)
    print("EDON v8 Generalization Test (Multiple Seeds)")
    print("="*70)
    print("\nTesting model on different seeds to verify generalization")
    print("Training seed: 0")
    print("Test seeds: 0 (training), 42, 100, 200")
    print("="*70)
    
    seeds = [0, 42, 100, 200]
    results = {}
    
    for seed in seeds:
        summary = evaluate_seed(seed, episodes=30)
        if summary:
            results[seed] = summary
    
    # Print comparison
    print("\n" + "="*70)
    print("GENERALIZATION RESULTS")
    print("="*70)
    
    baseline_int = 40.30
    baseline_stab = 0.0208
    
    for seed in seeds:
        if seed in results:
            r = results[seed]
            print(f"\nSeed {seed}:")
            print(f"  Interventions/ep: {r['interventions_per_episode']:.2f}")
            print(f"  Stability: {r['stability_avg']:.4f}")
            print(f"  EDON Score: {r['edon_score_v8']:.2f}")
            
            if seed != 0:  # Compare to baseline for non-training seeds
                delta_int = 100 * (baseline_int - r['interventions_per_episode']) / baseline_int if baseline_int > 0 else 0
                delta_stab = 100 * (baseline_stab - r['stability_avg']) / baseline_stab if baseline_stab > 0 else 0
                print(f"  vs Baseline: ΔInt={delta_int:+.1f}%, ΔStab={delta_stab:+.1f}%")
    
    # Check generalization
    print(f"\n{'='*70}")
    print("GENERALIZATION ASSESSMENT")
    print("="*70)
    
    training_seed = results.get(0, {})
    test_seeds = {s: r for s, r in results.items() if s != 0}
    
    if training_seed and test_seeds:
        train_int = training_seed['interventions_per_episode']
        train_stab = training_seed['stability_avg']
        
        test_ints = [r['interventions_per_episode'] for r in test_seeds.values()]
        test_stabs = [r['stability_avg'] for r in test_seeds.values()]
        
        avg_test_int = np.mean(test_ints)
        avg_test_stab = np.mean(test_stabs)
        
        print(f"\nTraining seed (0):")
        print(f"  Interventions/ep: {train_int:.2f}")
        print(f"  Stability: {train_stab:.4f}")
        
        print(f"\nTest seeds (avg):")
        print(f"  Interventions/ep: {avg_test_int:.2f}")
        print(f"  Stability: {avg_test_stab:.4f}")
        
        int_diff = abs(train_int - avg_test_int)
        stab_diff = abs(train_stab - avg_test_stab)
        
        print(f"\nDifference (training vs test avg):")
        print(f"  Interventions: {int_diff:.2f}")
        print(f"  Stability: {stab_diff:.4f}")
        
        if int_diff < 5 and stab_diff < 0.01:
            print(f"\n✅ GOOD GENERALIZATION: Performance similar across seeds")
        elif int_diff < 10 and stab_diff < 0.02:
            print(f"\n⚠️  MODERATE GENERALIZATION: Some variation across seeds")
        else:
            print(f"\n❌ POOR GENERALIZATION: Large variation suggests overfitting")
    
    print("="*70)
    
    # Save results
    output_path = Path("results/edon_v8_generalization_test.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[EVAL] ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise

