"""Show current evaluation results"""
import json
from pathlib import Path

print("="*70)
print("CURRENT EVALUATION RESULTS")
print("="*70)

# Load EDON v8 results
edon_path = Path("results/edon_v8_memory_features.json")
if edon_path.exists():
    with open(edon_path, 'r') as f:
        edon = json.load(f)
    
    seed = edon.get("seed", "N/A")
    run_metrics = edon.get("run_metrics", {})
    episodes_data = edon.get("episodes_data", edon.get("episodes", []))
    episodes_count = len(episodes_data) if isinstance(episodes_data, list) else 0
    
    print(f"\nSeed: {seed}")
    if seed == 42:
        print("⚠️  NOTE: This is from the old evaluation (seed=42)")
        print("   Waiting for new evaluation with seed=0 to complete...")
    elif seed == 0:
        print("✅ Seed matches training (seed=0)")
    
    print(f"Episodes: {episodes_count}")
    print(f"\nMetrics:")
    print(f"  Interventions/episode: {run_metrics.get('interventions_per_episode', 'N/A'):.2f}")
    print(f"  Stability (avg): {run_metrics.get('stability_avg', 'N/A'):.4f}")
    print(f"  Episode length (avg): {run_metrics.get('avg_episode_length', 'N/A'):.1f}")
    
    # Compare with baseline
    baseline_path = Path("results/baseline_high_stress_v44.json")
    if not baseline_path.exists():
        baseline_path = Path("results/baseline_high_stress.json")
    
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
        
        baseline_int = baseline.get("interventions_per_episode", 
                                    baseline.get("summary", {}).get("interventions_per_episode", 0))
        baseline_stab = baseline.get("stability_avg", 
                                    baseline.get("summary", {}).get("stability", 0))
        baseline_len = baseline.get("avg_episode_length", 
                                   baseline.get("summary", {}).get("avg_episode_length", 0))
        
        edon_int = run_metrics.get("interventions_per_episode", 0)
        edon_stab = run_metrics.get("stability_avg", 0)
        edon_len = run_metrics.get("avg_episode_length", 0)
        
        if baseline_int > 0:
            delta_int = 100 * (baseline_int - edon_int) / baseline_int
            delta_stab = 100 * (baseline_stab - edon_stab) / baseline_stab if baseline_stab > 0 else 0
            delta_len = 100 * (edon_len - baseline_len) / baseline_len if baseline_len > 0 else 0
            
            print(f"\n{'='*70}")
            print("COMPARISON WITH BASELINE")
            print("="*70)
            print(f"\nBaseline (seed may differ):")
            print(f"  Interventions/ep: {baseline_int:.2f}")
            print(f"  Stability: {baseline_stab:.4f}")
            print(f"  Episode length: {baseline_len:.1f}")
            
            print(f"\nEDON v8 Memory+Features (seed={seed}):")
            print(f"  Interventions/ep: {edon_int:.2f}")
            print(f"  Stability: {edon_stab:.4f}")
            print(f"  Episode length: {edon_len:.1f}")
            
            print(f"\nDelta:")
            print(f"  ΔInterventions: {delta_int:+.2f}% {'✅' if delta_int >= 10 else '❌'} (target: ≥10%)")
            print(f"  ΔStability: {delta_stab:+.2f}% {'✅' if abs(delta_stab) <= 5 else '❌'} (target: ±5%)")
            print(f"  ΔEpisode Length: {delta_len:+.2f}%")
            
            if delta_int >= 10 and abs(delta_stab) <= 5:
                print(f"\n✅ GOAL ACHIEVED: ≥10% intervention reduction with stable stability!")
            else:
                print(f"\n⚠️  Goal not fully met yet.")
                if seed == 42:
                    print(f"   Note: Re-evaluating with seed=0 may improve results.")
    else:
        print("\n⚠️  Baseline file not found for comparison")
    
    print("="*70)
else:
    print("Results file not found. Evaluation may not have run yet.")
