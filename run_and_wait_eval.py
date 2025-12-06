"""Run evaluation and wait for completion"""
import subprocess
import json
import time
from pathlib import Path

print("="*70)
print("Starting evaluation with seed=0...")
print("="*70)

# Start evaluation
results_file = Path("results/edon_v8_memory_features.json")
process = subprocess.Popen(
    ["python", "eval_v8_memory_features.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

print("Evaluation started. Monitoring progress...\n")

# Monitor output and file
last_episodes = 0
while True:
    # Check process
    if process.poll() is not None:
        # Process finished
        output, _ = process.communicate()
        if output:
            print(output)
        break
    
    # Check results file
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            seed = data.get("seed")
            episodes_data = data.get("episodes_data", data.get("episodes", []))
            episodes_count = len(episodes_data) if isinstance(episodes_data, list) else 0
            
            if episodes_count > last_episodes:
                print(f"Progress: {episodes_count}/30 episodes (seed={seed})")
                last_episodes = episodes_count
            
            if seed == 0 and episodes_count >= 30:
                print("\n✅ Evaluation complete!")
                process.terminate()
                break
        except:
            pass
    
    time.sleep(2)

# Read final results
if results_file.exists():
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    seed = data.get("seed")
    run_metrics = data.get("run_metrics", {})
    episodes_data = data.get("episodes_data", data.get("episodes", []))
    episodes_count = len(episodes_data) if isinstance(episodes_data, list) else 0
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Seed: {seed}")
    print(f"Episodes: {episodes_count}")
    print(f"Interventions/episode: {run_metrics.get('interventions_per_episode', 'N/A'):.2f}")
    print(f"Stability (avg): {run_metrics.get('stability_avg', 'N/A'):.4f}")
    print(f"Episode length (avg): {run_metrics.get('avg_episode_length', 'N/A'):.1f}")
    
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

