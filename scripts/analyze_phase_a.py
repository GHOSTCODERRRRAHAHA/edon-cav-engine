"""
Analyze Phase A training results to determine if Phase B is warranted.
"""

import sys
from pathlib import Path
import subprocess
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.compare_v8_vs_baseline import load_results, extract_metrics
from metrics.edon_v8_metrics import compute_episode_score_v8


def run_quick_eval(model_name: str, episodes: int = 30):
    """Run quick evaluation."""
    print(f"\nEvaluating {model_name}...")
    
    # Update run_eval to use this model temporarily
    # For now, just run eval
    cmd = [
        "python", "run_eval.py",
        "--mode", "edon",
        "--profile", "high_stress",
        "--episodes", str(episodes),
        "--seed", "42",
        "--output", f"results/{model_name}_eval.json",
        "--edon-gain", "1.0",
        "--edon-arch", "v8_strategy",
        "--edon-score"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def analyze_phase_a():
    """Analyze Phase A results."""
    print("="*80)
    print("Phase A Analysis: Fast Probing (100 episodes)")
    print("="*80)
    
    # Check if model exists (try v2 first, then v1)
    model_path = Path("models/edon_v8_strategy_phase_a_v2.pt")
    if not model_path.exists():
        model_path = Path("models/edon_v8_strategy_phase_a.pt")
    if not model_path.exists():
        print("ERROR: Phase A model not found!")
        return False
    
    print(f"\n[1] Model Status: [OK] Found ({model_path.stat().st_size / 1024:.1f} KB)")
    
    # Run baseline eval if needed
    baseline_file = Path("results/baseline_phase_a.json")
    if not baseline_file.exists():
        print("\n[2] Running baseline evaluation...")
        cmd = [
            "python", "run_eval.py",
            "--mode", "baseline",
            "--profile", "high_stress",
            "--episodes", "30",
            "--seed", "42",
            "--output", str(baseline_file),
            "--edon-score"
        ]
        subprocess.run(cmd)
    
    # Run Phase A eval
    if "v2" in str(model_path):
        phase_a_file = Path("results/edon_v8_strategy_phase_a_v2_eval.json")
        temp_model_name = "edon_v8_strategy_intervention_first_v2"
    else:
        phase_a_file = Path("results/edon_v8_strategy_phase_a_eval.json")
        temp_model_name = "edon_v8_strategy_intervention_first"
    if not phase_a_file.exists():
        print("\n[3] Running Phase A model evaluation...")
        # Temporarily copy model to expected name
        import shutil
        temp_model = Path("models/edon_v8_strategy_intervention_first.pt")
        if temp_model.exists():
            temp_model.unlink()
        shutil.copy(model_path, temp_model)
        
        cmd = [
            "python", "run_eval.py",
            "--mode", "edon",
            "--profile", "high_stress",
            "--episodes", "30",
            "--seed", "42",
            "--output", str(phase_a_file),
            "--edon-gain", "1.0",
            "--edon-arch", "v8_strategy",
            "--edon-score"
        ]
        subprocess.run(cmd)
    
    # Load and compare
    if baseline_file.exists() and phase_a_file.exists():
        print("\n[4] Comparing results...")
        baseline_data = load_results(str(baseline_file))
        phase_a_data = load_results(str(phase_a_file))
        
        baseline_metrics = extract_metrics(baseline_data)
        phase_a_metrics = extract_metrics(phase_a_data)
        
        # Compute EDON scores
        baseline_run = baseline_data.get("run_metrics", {})
        baseline_summary = {
            "interventions_per_episode": baseline_run.get("interventions_per_episode", 0),
            "stability_avg": baseline_run.get("stability_avg", 0.0),
            "avg_episode_length": baseline_run.get("avg_episode_length", 0)
        }
        baseline_score = compute_episode_score_v8(baseline_summary)
        
        phase_a_run = phase_a_data.get("run_metrics", {})
        phase_a_summary = {
            "interventions_per_episode": phase_a_run.get("interventions_per_episode", 0),
            "stability_avg": phase_a_run.get("stability_avg", 0.0),
            "avg_episode_length": phase_a_run.get("avg_episode_length", 0)
        }
        phase_a_score = compute_episode_score_v8(phase_a_summary)
        
        # Print comparison
        print("\n" + "="*80)
        print("Phase A Results vs Baseline")
        print("="*80)
        print(f"{'Metric':<30} {'Baseline':<15} {'Phase A':<15} {'Delta%':<15}")
        print("-"*80)
        
        baseline_int = baseline_metrics.get("interventions", 0.0)
        phase_a_int = phase_a_metrics.get("interventions", 0.0)
        int_delta = 100.0 * (phase_a_int - baseline_int) / baseline_int if baseline_int > 0 else 0.0
        print(f"{'Interventions/ep':<30} {baseline_int:<15.2f} {phase_a_int:<15.2f} {int_delta:>+6.1f}%")
        
        baseline_stab = baseline_metrics.get("stability", 0.0)
        phase_a_stab = phase_a_metrics.get("stability", 0.0)
        stab_delta = 100.0 * (phase_a_stab - baseline_stab) / baseline_stab if baseline_stab > 0 else 0.0
        print(f"{'Stability':<30} {baseline_stab:<15.4f} {phase_a_stab:<15.4f} {stab_delta:>+6.1f}%")
        
        print(f"{'EDON Score':<30} {baseline_score:<15.2f} {phase_a_score:<15.2f} {(phase_a_score - baseline_score):>+6.2f}")
        
        print("\n" + "="*80)
        print("Phase A Assessment:")
        print("-"*80)
        
        # Check learning indicators
        learning_ok = True
        if abs(int_delta) < 1.0:
            print("[WARNING] Interventions haven't budged - consider increasing w_intervention")
            learning_ok = False
        else:
            print(f"[OK] Interventions moving ({int_delta:+.1f}%)")
        
        if abs(stab_delta) > 10.0:
            print(f"[WARNING] Stability degrading ({stab_delta:+.1f}%) - consider increasing w_stability")
            learning_ok = False
        else:
            print(f"[OK] Stability acceptable ({stab_delta:+.1f}%)")
        
        if phase_a_score < baseline_score - 2.0:
            print(f"[WARNING] EDON Score regressing ({phase_a_score - baseline_score:+.2f})")
            learning_ok = False
        else:
            print(f"[OK] EDON Score acceptable ({phase_a_score - baseline_score:+.2f})")
        
        print("\n" + "="*80)
        if learning_ok:
            print("[PASS] PHASE A PASSED - Proceed to Phase B (200-400 episodes)")
            return True
        else:
            print("[FAIL] PHASE A FAILED - Tune weights before Phase B")
            print("\nRecommendations:")
            if abs(int_delta) < 1.0:
                print("  - Increase --w-intervention (try 7.0 or 10.0)")
            if abs(stab_delta) > 10.0:
                print("  - Increase --w-stability (try 1.5 or 2.0)")
                print("  - Increase --w-stability-episode (try 15.0 or 20.0)")
            return False
    
    return False


if __name__ == "__main__":
    analyze_phase_a()

