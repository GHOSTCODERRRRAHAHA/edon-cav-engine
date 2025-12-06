#!/usr/bin/env python3
"""
Full EDON Evaluation Runner

Runs complete A/B test: baseline vs EDON with real humanoid environment.
"""

import sys
import subprocess
import requests
from pathlib import Path
import json

# EDON server configuration
EDON_BASE_URL = "http://127.0.0.1:8001"
HEALTH_URL = f"{EDON_BASE_URL}/health"


def check_server_health() -> bool:
    """Check if EDON v2 server is running and healthy."""
    try:
        response = requests.get(HEALTH_URL, timeout=2.0)
        if response.status_code == 200:
            health_data = response.json()
            if health_data.get("mode") == "v2":
                print(f"[OK] EDON v2 server is running (mode: {health_data.get('mode')})")
                return True
            else:
                print(f"[WARNING] Server is running but not in v2 mode (mode: {health_data.get('mode', 'unknown')})")
                return False
        else:
            print(f"[ERROR] Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] EDON v2 server is not running on {EDON_BASE_URL}")
        print("  Please run: python start_edon_v2_server.py")
        return False
    except Exception as e:
        print(f"[ERROR] Health check failed: {e}")
        return False


def run_evaluation(mode: str, episodes: int, output: str, seed: int = 42) -> bool:
    """Run evaluation for a given mode."""
    print(f"\n{'='*70}")
    print(f"Running {mode.upper()} evaluation ({episodes} episodes)")
    print(f"{'='*70}\n")
    
    cmd = [
        sys.executable,
        "run_eval.py",
        "--mode", mode,
        "--episodes", str(episodes),
        "--output", output,
        "--seed", str(seed),
        "--randomize-env"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Evaluation failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to run evaluation: {e}")
        return False


def generate_plots(baseline_file: str, edon_file: str, output_dir: str) -> bool:
    """Generate comparison plots."""
    print(f"\n{'='*70}")
    print("Generating comparison plots")
    print(f"{'='*70}\n")
    
    cmd = [
        sys.executable,
        "plot_results.py",
        "--baseline", baseline_file,
        "--edon", edon_file,
        "--output", output_dir
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Plot generation failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to generate plots: {e}")
        return False


def compute_improvements(baseline_file: str, edon_file: str) -> dict:
    """Compute percentage improvements from results."""
    try:
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
        with open(edon_file, 'r') as f:
            edon = json.load(f)
        
        improvements = {}
        
        # Interventions (lower is better)
        baseline_interventions = baseline["interventions_per_episode"]
        edon_interventions = edon["interventions_per_episode"]
        if baseline_interventions > 0:
            improvements["intervention_reduction_pct"] = (
                (baseline_interventions - edon_interventions) / baseline_interventions * 100
            )
        else:
            improvements["intervention_reduction_pct"] = 0.0
        
        # Freeze events (lower is better)
        baseline_freezes = baseline["freeze_events_per_episode"]
        edon_freezes = edon["freeze_events_per_episode"]
        if baseline_freezes > 0:
            improvements["freeze_reduction_pct"] = (
                (baseline_freezes - edon_freezes) / baseline_freezes * 100
            )
        else:
            improvements["freeze_reduction_pct"] = 0.0
        
        # Stability (lower is better)
        baseline_stability = baseline["stability_avg"]
        edon_stability = edon["stability_avg"]
        if baseline_stability > 0:
            improvements["stability_improvement_pct"] = (
                (baseline_stability - edon_stability) / baseline_stability * 100
            )
        else:
            improvements["stability_improvement_pct"] = 0.0
        
        # Success rate (higher is better)
        baseline_success = baseline["success_rate"]
        edon_success = edon["success_rate"]
        if baseline_success > 0:
            improvements["success_improvement_pct"] = (
                (edon_success - baseline_success) / baseline_success * 100
            )
        else:
            improvements["success_improvement_pct"] = (
                edon_success * 100 if edon_success > 0 else 0.0
            )
        
        return improvements
    except Exception as e:
        print(f"[ERROR] Failed to compute improvements: {e}")
        return {}


def main():
    """Main evaluation runner."""
    print("=" * 70)
    print("EDON Full Evaluation Runner")
    print("=" * 70)
    print()
    
    # Check server health
    print("Checking EDON v2 server health...")
    if not check_server_health():
        print("\n[ERROR] Cannot proceed without EDON v2 server.")
        print("  Please start the server: python start_edon_v2_server.py")
        sys.exit(1)
    
    print()
    
    # Configuration
    episodes = 20
    seed = 42
    baseline_file = "results/baseline_real.json"
    edon_file = "results/edon_real.json"
    plots_dir = "plots_real"
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    Path(plots_dir).mkdir(exist_ok=True)
    
    # Run baseline evaluation
    print(f"Starting baseline evaluation ({episodes} episodes)...")
    if not run_evaluation("baseline", episodes, baseline_file, seed):
        print("[ERROR] Baseline evaluation failed")
        sys.exit(1)
    
    # Run EDON evaluation
    print(f"\nStarting EDON evaluation ({episodes} episodes)...")
    if not run_evaluation("edon", episodes, edon_file, seed):
        print("[ERROR] EDON evaluation failed")
        sys.exit(1)
    
    # Generate plots
    print(f"\nGenerating comparison plots...")
    if not generate_plots(baseline_file, edon_file, plots_dir):
        print("[WARNING] Plot generation failed, but continuing...")
    
    # Compute and print improvements
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS SUMMARY")
    print(f"{'='*70}\n")
    
    improvements = compute_improvements(baseline_file, edon_file)
    
    if improvements:
        print("EDON vs Baseline Improvements:")
        print("-" * 70)
        print(f"Intervention reduction: {improvements.get('intervention_reduction_pct', 0.0):.1f}%")
        print(f"Freeze reduction: {improvements.get('freeze_reduction_pct', 0.0):.1f}%")
        print(f"Stability improvement: {improvements.get('stability_improvement_pct', 0.0):.1f}%")
        print(f"Success rate improvement: {improvements.get('success_improvement_pct', 0.0):.1f}%")
        print("-" * 70)
    else:
        print("[WARNING] Could not compute improvements")
    
    print(f"\nResults saved to:")
    print(f"  Baseline: {baseline_file}")
    print(f"  EDON: {edon_file}")
    print(f"  Plots: {plots_dir}/")
    print()
    print("=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

