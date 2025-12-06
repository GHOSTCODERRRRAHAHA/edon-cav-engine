#!/usr/bin/env python3
"""
Analyze and compare results from comprehensive test suite.
Computes percentage improvements for interventions and stability.
"""

import json
from pathlib import Path
from typing import Dict, Optional

STRESS_PROFILES = ["normal_stress", "high_stress", "hell_stress"]
EDON_GAINS = [0.60, 0.75, 0.90, 1.00]

results_dir = Path("results")

def load_metrics(filepath: Path) -> Optional[Dict]:
    """Load metrics from a result file."""
    if not filepath.exists():
        return None
    try:
        with open(filepath) as f:
            data = json.load(f)
            # Try different JSON structures
            if "run_metrics" in data:
                return data["run_metrics"]
            elif "interventions_per_episode" in data:
                # Direct metrics structure
                return data
            elif "metrics" in data:
                return data["metrics"]
    except Exception as e:
        print(f"Error loading {filepath.name}: {e}")
    return None

def compute_improvement(baseline_value: float, edon_value: float, lower_is_better: bool = True) -> float:
    """Compute percentage improvement."""
    if baseline_value == 0:
        return 0.0
    if lower_is_better:
        # For interventions, stability (lower is better)
        improvement = ((baseline_value - edon_value) / baseline_value) * 100
    else:
        # For success rate (higher is better)
        improvement = ((edon_value - baseline_value) / baseline_value) * 100
    return improvement

def main():
    print("="*70)
    print("EDON PERFORMANCE ANALYSIS")
    print("="*70)
    
    all_results = {}
    
    # Load baseline results
    print("\nLoading baseline results...")
    baselines = {}
    for profile in STRESS_PROFILES:
        filepath = results_dir / f"baseline_{profile}_v44.json"
        metrics = load_metrics(filepath)
        if metrics:
            # Handle different metric field names
            num_episodes = metrics.get("num_episodes") or metrics.get("episodes", 1)
            total_int = metrics.get("total_interventions") or metrics.get("interventions_total", 0)
            mean_stab = metrics.get("mean_stability_score") or metrics.get("stability_avg", 0.0)
            total_freeze = metrics.get("total_freeze_events") or metrics.get("freeze_events_total", 0)
            
            baselines[profile] = {
                "interventions": total_int / num_episodes if num_episodes > 0 else 0,
                "stability": mean_stab,
                "freezes": total_freeze / num_episodes if num_episodes > 0 else 0,
            }
            print(f"  [OK] {profile}: {baselines[profile]['interventions']:.1f} interventions/ep, stability={baselines[profile]['stability']:.4f}")
        else:
            print(f"  [MISSING] {profile}: Not found")
    
    # Load EDON results and compute improvements
    print("\nLoading EDON results and computing improvements...")
    print("-" * 70)
    
    for profile in STRESS_PROFILES:
        if profile not in baselines:
            continue
        
        print(f"\n{profile.upper()}:")
        print(f"  Baseline: {baselines[profile]['interventions']:.1f} interventions/ep, stability={baselines[profile]['stability']:.4f}")
        print()
        
        best_gain = None
        best_improvement = -999
        
        for gain in EDON_GAINS:
            tag = f"{int(gain * 100):03d}"
            filepath = results_dir / f"edon_{profile}_v44_g{tag}.json"
            metrics = load_metrics(filepath)
            
            if metrics:
                # Handle different metric field names
                num_episodes = metrics.get("num_episodes") or metrics.get("episodes", 1)
                total_int = metrics.get("total_interventions") or metrics.get("interventions_total", 0)
                mean_stab = metrics.get("mean_stability_score") or metrics.get("stability_avg", 0.0)
                total_freeze = metrics.get("total_freeze_events") or metrics.get("freeze_events_total", 0)
                
                edon_interventions = total_int / num_episodes if num_episodes > 0 else 0
                edon_stability = mean_stab
                edon_freezes = total_freeze / num_episodes if num_episodes > 0 else 0
                
                int_improvement = compute_improvement(
                    baselines[profile]["interventions"], 
                    edon_interventions, 
                    lower_is_better=True
                )
                stab_improvement = compute_improvement(
                    baselines[profile]["stability"], 
                    edon_stability, 
                    lower_is_better=True
                )
                freeze_improvement = compute_improvement(
                    baselines[profile]["freezes"], 
                    edon_freezes, 
                    lower_is_better=True
                )
                
                total_improvement = (int_improvement + stab_improvement) / 2
                
                if total_improvement > best_improvement:
                    best_improvement = total_improvement
                    best_gain = gain
                
                # Color coding (using text indicators)
                int_indicator = "[GOOD]" if int_improvement >= 5 else "[OK]" if int_improvement >= 0 else "[BAD]"
                stab_indicator = "[GOOD]" if stab_improvement >= 5 else "[OK]" if stab_improvement >= 0 else "[BAD]"
                
                print(f"  Gain {gain:.2f}:")
                print(f"    Interventions: {edon_interventions:.1f}/ep  {int_indicator} {int_improvement:+.1f}%")
                print(f"    Stability:     {edon_stability:.4f}      {stab_indicator} {stab_improvement:+.1f}%")
                print(f"    Freezes:       {edon_freezes:.2f}/ep     {freeze_improvement:+.1f}%")
            else:
                print(f"  Gain {gain:.2f}: [...] Not completed")
        
        if best_gain is not None:
            print(f"\n  [BEST] Best gain for {profile}: {best_gain:.2f} ({best_improvement:.1f}% improvement)")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nTarget: 5-15%+ improvements across scenarios")
    print("Check results above to see if targets are met.")

if __name__ == "__main__":
    main()

