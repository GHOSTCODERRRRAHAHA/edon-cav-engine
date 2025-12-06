#!/usr/bin/env python3
"""
EDON Configuration Sweep Script

Runs multiple EDON evaluations with random configurations and identifies the best performing configs.
"""

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Configuration
N_RUNS = 50  # Number of random configs to test
BASELINE_FILE = "results/baseline_high_30.json"
BASELINE_PROFILE = "high_stress"
BASELINE_EPISODES = 30
BASELINE_SEED = 42

# Parameter ranges for random sampling
EDON_GAIN_RANGE = (0.3, 1.3)
GAIN_STABLE_RANGE = (0.5, 1.1)
GAIN_WARNING_RANGE = (0.8, 1.3)
CLAMP_RATIO_STABLE_RANGE = (1.05, 1.30)
CLAMP_RATIO_WARNING_RANGE = (1.10, 1.60)


def load_json(filepath: str) -> Dict:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_metrics(data: Dict) -> Tuple[float, float, float]:
    """Extract interventions, stability, and episode length from results."""
    if 'run_metrics' in data:
        # New format
        rm = data['run_metrics']
        return (
            rm.get('interventions_per_episode', 0.0),
            rm.get('stability_avg', 0.0),
            rm.get('avg_episode_length', 0.0)
        )
    else:
        # Old format
        return (
            data.get('interventions_per_episode', 0.0),
            data.get('stability_avg', 0.0),
            data.get('avg_episode_length', 0.0)
        )


def compute_metrics(baseline_data: Dict, edon_data: Dict) -> Tuple[float, float, float]:
    """
    Compute ΔInterventions%, ΔStability%, and Average%.
    
    Returns:
        (delta_interventions_pct, delta_stability_pct, average_pct)
    """
    I_b, S_b, _ = extract_metrics(baseline_data)
    I_e, S_e, _ = extract_metrics(edon_data)
    
    # Lower is better for both interventions and stability
    # So positive delta = improvement
    delta_interventions_pct = 100 * (I_b - I_e) / I_b if I_b > 0 else 0.0
    delta_stability_pct = 100 * (S_b - S_e) / S_b if S_b > 0 else 0.0
    average_pct = (delta_interventions_pct + delta_stability_pct) / 2.0
    
    return delta_interventions_pct, delta_stability_pct, average_pct


def ensure_baseline():
    """Ensure baseline exists, run if needed."""
    if Path(BASELINE_FILE).exists():
        print(f"[OK] Baseline exists: {BASELINE_FILE}")
        return
    
    print(f"[INFO] Baseline not found. Running baseline evaluation...")
    cmd = [
        "python", "run_eval.py",
        "--mode", "baseline",
        "--profile", BASELINE_PROFILE,
        "--episodes", str(BASELINE_EPISODES),
        "--seed", str(BASELINE_SEED),
        "--output", BASELINE_FILE
    ]
    
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] Baseline run failed:")
        print(result.stderr)
        sys.exit(1)
    
    if not Path(BASELINE_FILE).exists():
        print(f"[ERROR] Baseline file not created: {BASELINE_FILE}")
        sys.exit(1)
    
    print(f"[OK] Baseline created: {BASELINE_FILE}")


def generate_random_config(run_id: int) -> Dict:
    """Generate a random EDON configuration."""
    return {
        "GAIN_STABLE": random.uniform(*GAIN_STABLE_RANGE),
        "GAIN_WARNING": random.uniform(*GAIN_WARNING_RANGE),
        "GAIN_RECOVERY": 1.1,  # Fixed for now
        "CLAMP_RATIO_STABLE": random.uniform(*CLAMP_RATIO_STABLE_RANGE),
        "CLAMP_RATIO_WARNING": random.uniform(*CLAMP_RATIO_WARNING_RANGE),
        "CLAMP_RATIO_RECOVERY": 1.50,  # Fixed for now
        "W_PREFALL_STABLE": 0.3,  # Fixed for now
        "W_SAFE_STABLE": 0.7,  # Fixed for now
        "W_PREFALL_WARNING": 0.5,  # Fixed for now
        "W_SAFE_WARNING": 0.5,  # Fixed for now
        "W_PREFALL_RECOVERY": 0.5,  # Fixed for now
        "W_SAFE_RECOVERY": 0.5,  # Fixed for now
    }


def run_edon_evaluation(run_id: int, edon_gain: float, config: Dict) -> bool:
    """Run a single EDON evaluation with given config."""
    # Write config file
    config_file = f"edon_cfg_run_{run_id}.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Set environment variable
    env = os.environ.copy()
    env["EDON_CONFIG_PATH"] = config_file
    
    # Run evaluation
    output_file = f"results/edon_high_run_{run_id}.json"
    cmd = [
        "python", "run_eval.py",
        "--mode", "edon",
        "--profile", BASELINE_PROFILE,
        "--edon-gain", str(edon_gain),
        "--episodes", str(BASELINE_EPISODES),
        "--seed", str(BASELINE_SEED),
        "--output", output_file
    ]
    
    print(f"[RUN {run_id:3d}] edon_gain={edon_gain:.2f}, G_STABLE={config['GAIN_STABLE']:.2f}, G_WARN={config['GAIN_WARNING']:.2f}, C_STABLE={config['CLAMP_RATIO_STABLE']:.2f}, C_WARN={config['CLAMP_RATIO_WARNING']:.2f}")
    
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    # Clean up config file
    if Path(config_file).exists():
        os.remove(config_file)
    
    if result.returncode != 0:
        print(f"[ERROR] Run {run_id} failed:")
        print(result.stderr[:500])  # Truncate long errors
        return False
    
    if not Path(output_file).exists():
        print(f"[ERROR] Output file not created: {output_file}")
        return False
    
    return True


def main():
    """Main sweep execution."""
    print("="*70)
    print("EDON CONFIGURATION SWEEP")
    print("="*70)
    print(f"Runs: {N_RUNS}")
    print(f"Baseline: {BASELINE_FILE}")
    print(f"Profile: {BASELINE_PROFILE}, Episodes: {BASELINE_EPISODES}, Seed: {BASELINE_SEED}")
    print("="*70)
    
    # Ensure baseline exists
    ensure_baseline()
    
    # Load baseline metrics
    baseline_data = load_json(BASELINE_FILE)
    I_b, S_b, L_b = extract_metrics(baseline_data)
    print(f"\n[BASELINE] Interventions: {I_b:.2f}, Stability: {S_b:.4f}, Length: {L_b:.1f}")
    print()
    
    # Ensure results directory exists
    Path("results").mkdir(exist_ok=True)
    
    # Run sweep
    results = []
    successful_runs = 0
    
    for run_id in range(N_RUNS):
        # Generate random config
        edon_gain = random.uniform(*EDON_GAIN_RANGE)
        config = generate_random_config(run_id)
        
        # Run evaluation
        if not run_edon_evaluation(run_id, edon_gain, config):
            continue
        
        successful_runs += 1
        
        # Load results and compute metrics
        try:
            edon_data = load_json(f"results/edon_high_run_{run_id}.json")
            dI, dS, avg = compute_metrics(baseline_data, edon_data)
            
            results.append({
                "run_id": run_id,
                "edon_gain": edon_gain,
                "GAIN_STABLE": config["GAIN_STABLE"],
                "GAIN_WARNING": config["GAIN_WARNING"],
                "CLAMP_RATIO_STABLE": config["CLAMP_RATIO_STABLE"],
                "CLAMP_RATIO_WARNING": config["CLAMP_RATIO_WARNING"],
                "delta_interventions": dI,
                "delta_stability": dS,
                "average": avg
            })
            
            print(f"         → ΔI={dI:+.1f}%, ΔS={dS:+.1f}%, Avg={avg:+.1f}%")
        except Exception as e:
            print(f"[ERROR] Failed to process results for run {run_id}: {e}")
            continue
    
    print(f"\n[INFO] Completed {successful_runs}/{N_RUNS} successful runs")
    
    if not results:
        print("[ERROR] No successful runs to analyze")
        sys.exit(1)
    
    # Sort by Average% descending
    results.sort(key=lambda x: x["average"], reverse=True)
    
    # Print top 5
    print("\n" + "="*70)
    print(f"TOP 5 EDON CONFIGS ({BASELINE_PROFILE}, {BASELINE_EPISODES} eps, baseline = {BASELINE_FILE})")
    print("="*70)
    print(f"{'run':<5} {'gain':<6} {'G_STABLE':<9} {'G_WARN':<8} {'C_STABLE':<9} {'C_WARN':<8} {'ΔI(%)':<8} {'ΔS(%)':<8} {'Avg(%)':<8}")
    print("-"*70)
    
    for i, r in enumerate(results[:5], 1):
        print(f"{r['run_id']:<5} {r['edon_gain']:<6.2f} {r['GAIN_STABLE']:<9.2f} {r['GAIN_WARNING']:<8.2f} "
              f"{r['CLAMP_RATIO_STABLE']:<9.2f} {r['CLAMP_RATIO_WARNING']:<8.2f} "
              f"{r['delta_interventions']:+7.1f} {r['delta_stability']:+7.1f} {r['average']:+7.1f}")
    
    # Print best config as JSON
    best = results[0]
    print("\n" + "="*70)
    print("BEST CONFIG:")
    print("="*70)
    best_config_json = {
        "edon_gain": best["edon_gain"],
        "GAIN_STABLE": best["GAIN_STABLE"],
        "GAIN_WARNING": best["GAIN_WARNING"],
        "GAIN_RECOVERY": 1.1,  # Fixed
        "CLAMP_RATIO_STABLE": best["CLAMP_RATIO_STABLE"],
        "CLAMP_RATIO_WARNING": best["CLAMP_RATIO_WARNING"],
        "CLAMP_RATIO_RECOVERY": 1.50,  # Fixed
        "W_PREFALL_STABLE": 0.3,  # Fixed
        "W_SAFE_STABLE": 0.7,  # Fixed
        "W_PREFALL_WARNING": 0.5,  # Fixed
        "W_SAFE_WARNING": 0.5,  # Fixed
        "W_PREFALL_RECOVERY": 0.5,  # Fixed
        "W_SAFE_RECOVERY": 0.5,  # Fixed
        "delta_interventions": round(best["delta_interventions"], 1),
        "delta_stability": round(best["delta_stability"], 1),
        "average": round(best["average"], 1)
    }
    print(json.dumps(best_config_json, indent=2))
    print("="*70)
    
    # Save best config to file
    best_config_file = "results/best_edon_config.json"
    with open(best_config_file, 'w') as f:
        json.dump(best_config_json, f, indent=2)
    print(f"\n[INFO] Best config saved to: {best_config_file}")


if __name__ == "__main__":
    main()

