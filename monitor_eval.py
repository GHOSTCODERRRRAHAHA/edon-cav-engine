"""Monitor evaluation until it completes"""
import json
import time
from pathlib import Path

results_file = Path("results/edon_v8_memory_features.json")
check_interval = 3  # Check every 3 seconds
max_checks = 200  # Max 10 minutes

print("Monitoring evaluation progress...")
print("="*70)

for i in range(max_checks):
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            seed = data.get("seed")
            episodes_data = data.get("episodes_data", data.get("episodes", []))
            episodes_count = len(episodes_data) if isinstance(episodes_data, list) else 0
            run_metrics = data.get("run_metrics", {})
            
            if seed == 0 and episodes_count >= 30:
                print("\n" + "="*70)
                print("✅ EVALUATION COMPLETE!")
                print("="*70)
                print(f"\nResults (seed=0, matching training):")
                print(f"  Episodes: {episodes_count}")
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
                    
                    edon_int = run_metrics.get("interventions_per_episode", 0)
                    edon_stab = run_metrics.get("stability_avg", 0)
                    
                    if baseline_int > 0:
                        delta_int = 100 * (baseline_int - edon_int) / baseline_int
                        delta_stab = 100 * (baseline_stab - edon_stab) / baseline_stab if baseline_stab > 0 else 0
                        
                        print(f"\n{'='*70}")
                        print("COMPARISON WITH BASELINE")
                        print("="*70)
                        print(f"Baseline:")
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
                        print("="*70)
                
                break
            elif seed == 0:
                if i % 10 == 0:  # Print every 30 seconds
                    print(f"[{i*check_interval}s] Progress: {episodes_count}/30 episodes...")
            elif seed == 42:
                if i == 0:
                    print("⚠️  Old results (seed=42) detected. Waiting for new evaluation...")
        except (json.JSONDecodeError, KeyError):
            # File might be partially written
            pass
        except Exception as e:
            if i % 20 == 0:
                print(f"Error reading file: {e}")
    else:
        if i == 0:
            print("Waiting for results file...")
    
    time.sleep(check_interval)
else:
    print(f"\n⚠️  Timeout after {max_checks * check_interval}s. Evaluation may still be running.")

