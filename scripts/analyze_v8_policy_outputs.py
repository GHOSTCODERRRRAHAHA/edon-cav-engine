"""Analyze what the v8 policy is actually outputting during evaluation."""

import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

def analyze_v8_outputs(result_file: str):
    """Analyze strategy and modulation outputs from v8 evaluation."""
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    episodes = data.get("episodes", [])
    
    print("="*80)
    print("v8 Policy Output Analysis")
    print("="*80)
    print()
    
    # Collect strategy selections
    strategies = []
    gain_scales = []
    lateral_compliances = []
    step_height_biases = []
    fail_risks = []
    
    for episode in episodes:
        # Check if episode has step-level data
        # Most likely, we need to check metadata or reconstruct from episode data
        metadata = episode.get("metadata", {})
        
        # Try to get strategy info from metadata
        if "strategy_id" in metadata:
            strategies.append(metadata["strategy_id"])
        if "gain_scale" in metadata:
            gain_scales.append(metadata["gain_scale"])
        if "lateral_compliance" in metadata:
            lateral_compliances.append(metadata["lateral_compliance"])
        if "fail_risk" in metadata:
            fail_risks.append(metadata["fail_risk"])
    
    # If no metadata, try to extract from episode summary
    if not strategies:
        print("No strategy data found in metadata. Checking episode structure...")
        print(f"Episode keys: {list(episodes[0].keys()) if episodes else 'No episodes'}")
        if episodes:
            print(f"Sample episode structure: {json.dumps(episodes[0], indent=2)[:500]}")
    
    if strategies:
        strategy_counts = Counter(strategies)
        print("Strategy Selection:")
        for strategy_id, count in strategy_counts.most_common():
            strategy_name = ["NORMAL", "HIGH_DAMPING", "RECOVERY_BALANCE", "COMPLIANT_TERRAIN"][strategy_id]
            print(f"  {strategy_name}: {count} ({100*count/len(strategies):.1f}%)")
        print()
    
    if gain_scales:
        print(f"Gain Scale: avg={sum(gain_scales)/len(gain_scales):.3f}, min={min(gain_scales):.3f}, max={max(gain_scales):.3f}")
    if lateral_compliances:
        print(f"Lateral Compliance: avg={sum(lateral_compliances)/len(lateral_compliances):.3f}, min={min(lateral_compliances):.3f}, max={max(lateral_compliances):.3f}")
    if fail_risks:
        print(f"Fail Risk: avg={sum(fail_risks)/len(fail_risks):.3f}, min={min(fail_risks):.3f}, max={max(fail_risks):.3f}")
    
    print()
    print("="*80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=str, default="results/edon_v8_improved_final.json", help="v8 evaluation result file")
    args = parser.parse_args()
    analyze_v8_outputs(args.result)

