"""Check evaluation results"""
import json
import time
from pathlib import Path

results_file = Path("results/edon_v8_memory_features.json")

if results_file.exists():
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    seed = data.get("seed", "N/A")
    run_metrics = data.get("run_metrics", {})
    interventions = run_metrics.get("interventions_per_episode", "N/A")
    stability = run_metrics.get("stability_avg", "N/A")
    episodes_count = len(data.get("episodes_data", data.get("episodes", [])))
    
    print("="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Seed: {seed}")
    print(f"Episodes completed: {episodes_count}")
    print(f"Interventions/episode: {interventions}")
    print(f"Stability (avg): {stability}")
    print("="*70)
    
    # Check if seed matches training
    if seed == 0:
        print("✓ Seed matches training (seed=0)")
    else:
        print(f"⚠ Seed mismatch: evaluation used {seed}, training used 0")
else:
    print("Results file not found yet. Evaluation may still be running.")

