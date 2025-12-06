import json
import os
from pathlib import Path
from datetime import datetime

results_file = Path("results/edon_v8_memory_features.json")

if results_file.exists():
    mtime = os.path.getmtime(results_file)
    age_seconds = os.path.getmtime(results_file) - os.path.getmtime(results_file)
    age_minutes = (os.path.getmtime(results_file) - os.path.getmtime(results_file)) / 60
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    seed = data.get("seed")
    episodes_data = data.get("episodes_data", data.get("episodes", []))
    episodes_count = len(episodes_data) if isinstance(episodes_data, list) else 0
    run_metrics = data.get("run_metrics", {})
    
    print("="*70)
    print("EVALUATION STATUS")
    print("="*70)
    print(f"File: {results_file}")
    print(f"Last modified: {datetime.fromtimestamp(mtime)}")
    print(f"Seed: {seed}")
    print(f"Episodes: {episodes_count}/30")
    
    if seed == 0 and episodes_count >= 30:
        print("\n✅ EVALUATION COMPLETE!")
        print(f"\nResults:")
        print(f"  Interventions/episode: {run_metrics.get('interventions_per_episode', 'N/A')}")
        print(f"  Stability (avg): {run_metrics.get('stability_avg', 'N/A')}")
        print(f"  Episode length (avg): {run_metrics.get('avg_episode_length', 'N/A')}")
    elif seed == 0:
        print(f"\n⏳ Evaluation in progress... ({episodes_count}/30 episodes)")
    elif seed == 42:
        print(f"\n⚠️  Old results detected (seed=42). New evaluation with seed=0 may be running...")
    else:
        print(f"\n⚠️  Unexpected seed: {seed}")
else:
    print("Results file not found. Evaluation may not have started yet.")

