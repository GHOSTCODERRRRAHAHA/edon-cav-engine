"""Wait for evaluation to complete and show results"""
import json
import time
from pathlib import Path
import subprocess
import sys

results_file = Path("results/edon_v8_memory_features.json")
max_wait_time = 600  # 10 minutes max
check_interval = 5  # Check every 5 seconds

print("="*70)
print("Waiting for evaluation to complete...")
print("="*70)
print(f"Checking every {check_interval} seconds for results file updates...")
print(f"Looking for seed=0 in results...")
print()

start_time = time.time()
last_size = 0
last_seed = None

while True:
    elapsed = time.time() - start_time
    
    # Check if results file exists and has been updated
    if results_file.exists():
        try:
            # Check file size first (quick check)
            current_size = results_file.stat().st_size
            
            # If file size changed, it might be updating
            if current_size != last_size:
                print(f"[{int(elapsed)}s] File size changed: {current_size} bytes (was {last_size})")
                last_size = current_size
                time.sleep(2)  # Wait a bit for file to finish writing
                continue
            
            # Try to read the file
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            seed = data.get("seed", None)
            run_metrics = data.get("run_metrics", {})
            episodes_data = data.get("episodes_data", data.get("episodes", []))
            episodes_count = len(episodes_data) if isinstance(episodes_data, list) else 0
            
            # Check if seed matches what we want (0)
            if seed == 0:
                # Check if we have all episodes (should be 30)
                if episodes_count >= 30:
                    print(f"\n{'='*70}")
                    print("EVALUATION COMPLETE!")
                    print(f"{'='*70}\n")
                    
                    interventions = run_metrics.get("interventions_per_episode", "N/A")
                    stability = run_metrics.get("stability_avg", "N/A")
                    length = run_metrics.get("avg_episode_length", "N/A")
                    
                    print(f"Seed: {seed} ✓ (matches training)")
                    print(f"Episodes: {episodes_count}")
                    print(f"Interventions/episode: {interventions}")
                    print(f"Stability (avg): {stability}")
                    print(f"Episode length (avg): {length}")
                    print(f"\n{'='*70}")
                    
                    # Compare with baseline
                    baseline_path = Path("results/baseline_high_stress_v44.json")
                    if not baseline_path.exists():
                        baseline_path = Path("results/baseline_high_stress.json")
                    
                    if baseline_path.exists():
                        print("\nCOMPARISON WITH BASELINE:")
                        print("-"*70)
                        with open(baseline_path, 'r') as f:
                            baseline = json.load(f)
                        
                        baseline_int = baseline.get("interventions_per_episode", 
                                                    baseline.get("summary", {}).get("interventions_per_episode", 0))
                        baseline_stab = baseline.get("stability_avg", 
                                                    baseline.get("summary", {}).get("stability", 0))
                        
                        if baseline_int > 0:
                            delta_int = 100 * (baseline_int - interventions) / baseline_int
                            delta_stab = 100 * (baseline_stab - stability) / baseline_stab if baseline_stab > 0 else 0
                            
                            print(f"Baseline:")
                            print(f"  Interventions/ep: {baseline_int:.2f}")
                            print(f"  Stability: {baseline_stab:.4f}")
                            print(f"\nEDON v8 Memory+Features:")
                            print(f"  Interventions/ep: {interventions:.2f}")
                            print(f"  Stability: {stability:.4f}")
                            print(f"\nDelta:")
                            print(f"  ΔInterventions: {delta_int:+.2f}% {'✅' if delta_int >= 10 else '❌'} (target: ≥10%)")
                            print(f"  ΔStability: {delta_stab:+.2f}% {'✅' if abs(delta_stab) <= 5 else '❌'} (target: ±5%)")
                            
                            if delta_int >= 10 and abs(delta_stab) <= 5:
                                print(f"\n✅ GOAL ACHIEVED: ≥10% intervention reduction with stable stability!")
                            else:
                                print(f"\n⚠️  Goal not fully met yet.")
                    
                    print(f"{'='*70}\n")
                    sys.exit(0)
                else:
                    if last_seed != seed or episodes_count % 5 == 0:
                        print(f"[{int(elapsed)}s] Seed={seed}, Episodes={episodes_count}/30...")
                        last_seed = seed
            else:
                if last_seed != seed:
                    print(f"[{int(elapsed)}s] Found seed={seed} (waiting for seed=0)...")
                    last_seed = seed
        except (json.JSONDecodeError, KeyError) as e:
            # File might be partially written
            if elapsed % 10 < check_interval:
                print(f"[{int(elapsed)}s] File being written (JSON error: {type(e).__name__})...")
        except Exception as e:
            print(f"[{int(elapsed)}s] Error reading file: {e}")
    else:
        if elapsed % 10 < check_interval:
            print(f"[{int(elapsed)}s] Waiting for results file to be created...")
    
    # Check if we've waited too long
    if elapsed > max_wait_time:
        print(f"\n⚠️  Timeout after {max_wait_time}s. Evaluation may still be running.")
        print("Check manually: python eval_v8_memory_features.py")
        sys.exit(1)
    
    time.sleep(check_interval)

