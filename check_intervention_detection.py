"""Check if interventions are being detected correctly in evaluation"""
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

# Load a recent evaluation result
results_file = Path("results/edon_v8_generalization_test.json")
if results_file.exists():
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print("="*70)
    print("CHECKING INTERVENTION DETECTION IN EVALUATION")
    print("="*70)
    
    for seed, metrics in data.items():
        print(f"\nSeed {seed}:")
        print(f"  Interventions/episode: {metrics.get('interventions_per_episode', 0):.2f}")
        print(f"  Interventions total: {metrics.get('interventions_total', 0)}")
        print(f"  Stability: {metrics.get('stability_avg', 0):.4f}")
        print(f"  Episode length: {metrics.get('avg_episode_length', 0):.1f}")
    
    # Check if we can find any episode data with interventions
    edon_file = Path("results/edon_v8_memory_features.json")
    if edon_file.exists():
        with open(edon_file, 'r') as f:
            edon_data = json.load(f)
        
        episodes_data = edon_data.get("episodes_data", edon_data.get("episodes", []))
        print(f"\n{'='*70}")
        print("DETAILED EPISODE ANALYSIS")
        print("="*70)
        print(f"Total episodes: {len(episodes_data)}")
        
        interventions_by_episode = []
        for i, ep in enumerate(episodes_data[:10]):  # Check first 10 episodes
            interventions = ep.get("interventions", 0)
            interventions_by_episode.append(interventions)
            if interventions > 0:
                print(f"  Episode {i}: {interventions} interventions")
        
        if all(x == 0 for x in interventions_by_episode):
            print("\n⚠️  WARNING: All checked episodes show 0 interventions")
            print("   This could mean:")
            print("   1. The policy is actually preventing all interventions (good!)")
            print("   2. Intervention detection is not working (bad!)")
            print("   3. The episodes are too short or conditions are too easy")
        
        print(f"\nInterventions in first 10 episodes: {interventions_by_episode}")
        print(f"Total interventions in first 10: {sum(interventions_by_episode)}")
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("Run verify_interventions.py to test intervention detection")
    print("with known bad actions to verify the detection system works.")
    print("="*70)

else:
    print("Results file not found. Run evaluation first.")

