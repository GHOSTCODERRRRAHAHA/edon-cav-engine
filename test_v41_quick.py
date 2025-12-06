#!/usr/bin/env python3
"""
Quick test for V4.1 Adaptive Gain System.

Runs 100 episodes total, split into batches of 30 for faster feedback.
"""

import subprocess
import json
import os
from pathlib import Path
from typing import Dict
import statistics

# Test configuration (V4.1 baseline)
CONFIG = {
    "name": "V4.1_quick",
    "base_gain": 0.50,
    "prefall_min": 0.15,
    "prefall_max": 0.60,
    "safe_threshold": 0.75,
    "safe_gain": 0.12,
}

# Test parameters
EPISODES_PER_BATCH = 30
N_BATCHES = 1  # Just 1 batch of 30 episodes
TOTAL_EPISODES = 30
PROFILE = "high_stress"
EDON_GAIN = 0.75
SEED = 42  # Fixed seed for consistency

RESULTS_DIR = Path("results/v41_quick")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def apply_config_to_code(config: Dict) -> None:
    """Apply test configuration to run_eval.py by modifying the code."""
    import re
    import shutil
    from pathlib import Path
    
    # Create backup first
    backup_file = Path("run_eval.py.backup")
    if Path("run_eval.py").exists() and Path("run_eval.py").stat().st_size > 0:
        shutil.copy2("run_eval.py", backup_file)
        print(f"    Created backup: {backup_file}")
    else:
        print(f"    WARNING: run_eval.py is empty or missing! Cannot proceed.")
        print(f"    Please restore run_eval.py from backup or git.")
        raise FileNotFoundError("run_eval.py is corrupted or missing")
    
    # Read current code with UTF-8 encoding
    try:
        with open("run_eval.py", "r", encoding="utf-8") as f:
            code = f.read()
        
        if len(code) == 0:
            raise ValueError("File is empty")
    except Exception as e:
        print(f"    ERROR reading file: {e}")
        # Restore from backup
        if backup_file.exists():
            shutil.copy2(backup_file, "run_eval.py")
            print(f"    Restored from backup")
        raise
    
    # Replace BASE_GAIN (handle Unicode arrow character)
    code = re.sub(
        r'BASE_GAIN = \d+\.\d+\s+#.*',
        f'BASE_GAIN = {config["base_gain"]:.2f}  # Was 0.40 -> better stability backbone',
        code
    )
    
    # Replace PREFALL_MIN
    code = re.sub(
        r'PREFALL_MIN = \d+\.\d+',
        f'PREFALL_MIN = {config["prefall_min"]:.2f}',
        code
    )
    
    # Replace PREFALL_MAX
    code = re.sub(
        r'PREFALL_MAX = \d+\.\d+',
        f'PREFALL_MAX = {config["prefall_max"]:.2f}',
        code
    )
    
    # Replace SAFE_THRESHOLD
    code = re.sub(
        r'catastrophic_risk > \d+\.\d+',
        f'catastrophic_risk > {config["safe_threshold"]:.2f}',
        code
    )
    
    # Replace SAFE_GAIN
    code = re.sub(
        r'SAFE_GAIN = \d+\.\d+',
        f'SAFE_GAIN = {config["safe_gain"]:.2f}',
        code
    )
    
    # Write back with UTF-8 encoding
    try:
        with open("run_eval.py", "w", encoding="utf-8") as f:
            f.write(code)
        
        # Verify write succeeded
        if Path("run_eval.py").stat().st_size == 0:
            raise ValueError("File write resulted in empty file")
        
        print(f"    Configuration applied successfully")
    except Exception as e:
        print(f"    ERROR writing file: {e}")
        # Restore from backup
        if backup_file.exists():
            shutil.copy2(backup_file, "run_eval.py")
            print(f"    Restored from backup")
        raise


def run_batch(mode: str, batch_num: int, episodes: int) -> str:
    """Run a single batch and return output file path."""
    output_file = RESULTS_DIR / f"{CONFIG['name']}_{mode}_batch{batch_num}.json"
    
    # Ensure directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Use forward slashes for cross-platform compatibility
    output_path = str(output_file).replace("\\", "/")
    
    cmd = [
        "python", "run_eval.py",
        "--mode", mode,
        "--episodes", str(episodes),
        "--profile", PROFILE,
        "--seed", str(SEED + batch_num),  # Different seed per batch
        "--output", output_path,
    ]
    
    if mode == "edon":
        cmd.extend(["--edon-gain", str(EDON_GAIN)])
    
    print(f"  Running {mode} batch {batch_num} ({episodes} episodes, seed {SEED + batch_num})...")
    print(f"    Output: {output_path}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"    ERROR: Return code {result.returncode}")
        if result.stderr:
            print(f"    STDERR: {result.stderr[:500]}")  # First 500 chars
        if result.stdout:
            print(f"    STDOUT: {result.stdout[-500:]}")  # Last 500 chars
        return None
    
    # Check if file was actually created (try both forward and backslash paths)
    if not output_file.exists():
        # Try with backslashes for Windows
        output_file_win = Path(str(output_file).replace("/", "\\"))
        if output_file_win.exists():
            return str(output_file_win)
        
        print(f"    WARNING: Output file not created: {output_file}")
        print(f"    Command: {' '.join(cmd)}")
        if result.stdout:
            print(f"    Last 200 chars of stdout: {result.stdout[-200:]}")
        return None
    
    print(f"    ✓ File created: {output_file}")
    return str(output_file)


def compute_improvement(baseline_file: str, edon_file: str) -> Dict[str, float]:
    """Compute improvement metrics from result files."""
    try:
        with open(baseline_file, "r") as f:
            baseline_data = json.load(f)
        with open(edon_file, "r") as f:
            edon_data = json.load(f)
        
        baseline_metrics = baseline_data.get("run_metrics", {})
        edon_metrics = edon_data.get("run_metrics", {})
        
        # Compute percentage improvements
        baseline_interventions = baseline_metrics.get("interventions", 0)
        edon_interventions = edon_metrics.get("interventions", 0)
        interventions_improvement = (
            ((baseline_interventions - edon_interventions) / baseline_interventions * 100)
            if baseline_interventions > 0 else 0.0
        )
        
        baseline_stability = baseline_metrics.get("stability_score", 1.0)
        edon_stability = edon_metrics.get("stability_score", 1.0)
        stability_improvement = (
            ((baseline_stability - edon_stability) / baseline_stability * 100)
            if baseline_stability > 0 else 0.0
        )
        
        # Average improvement
        average_improvement = (interventions_improvement + stability_improvement) / 2.0
        
        return {
            "interventions": interventions_improvement,
            "stability": stability_improvement,
            "average": average_improvement,
            "baseline_interventions": baseline_interventions,
            "edon_interventions": edon_interventions,
            "baseline_stability": baseline_stability,
            "edon_stability": edon_stability,
        }
    except Exception as e:
        print(f"    ERROR computing improvement: {e}")
        return {
            "interventions": 0.0,
            "stability": 0.0,
            "average": 0.0,
            "baseline_interventions": 0,
            "edon_interventions": 0,
            "baseline_stability": 1.0,
            "edon_stability": 1.0,
        }


def aggregate_results(batch_results: list) -> Dict:
    """Aggregate results from multiple batches."""
    if not batch_results:
        return {
            "total_episodes": 0,
            "interventions": {"mean": 0.0, "std": 0.0},
            "stability": {"mean": 0.0, "std": 0.0},
            "average": {"mean": 0.0, "std": 0.0},
        }
    
    # Aggregate raw metrics
    total_baseline_interventions = sum(r["baseline_interventions"] for r in batch_results)
    total_edon_interventions = sum(r["edon_interventions"] for r in batch_results)
    total_baseline_stability = sum(r["baseline_stability"] for r in batch_results)
    total_edon_stability = sum(r["edon_stability"] for r in batch_results)
    
    # Overall improvements
    overall_interventions_improvement = (
        ((total_baseline_interventions - total_edon_interventions) / total_baseline_interventions * 100)
        if total_baseline_interventions > 0 else 0.0
    )
    overall_stability_improvement = (
        ((total_baseline_stability - total_edon_stability) / total_baseline_stability * 100)
        if total_baseline_stability > 0 else 0.0
    )
    overall_average = (overall_interventions_improvement + overall_stability_improvement) / 2.0
    
    # Per-batch statistics
    interventions_improvements = [r["interventions"] for r in batch_results]
    stability_improvements = [r["stability"] for r in batch_results]
    average_improvements = [r["average"] for r in batch_results]
    
    return {
        "total_episodes": len(batch_results) * EPISODES_PER_BATCH,
        "overall": {
            "interventions": overall_interventions_improvement,
            "stability": overall_stability_improvement,
            "average": overall_average,
        },
        "per_batch": {
            "interventions": {
                "mean": statistics.mean(interventions_improvements),
                "std": statistics.stdev(interventions_improvements) if len(interventions_improvements) > 1 else 0.0,
                "min": min(interventions_improvements),
                "max": max(interventions_improvements),
            },
            "stability": {
                "mean": statistics.mean(stability_improvements),
                "std": statistics.stdev(stability_improvements) if len(stability_improvements) > 1 else 0.0,
                "min": min(stability_improvements),
                "max": max(stability_improvements),
            },
            "average": {
                "mean": statistics.mean(average_improvements),
                "std": statistics.stdev(average_improvements) if len(average_improvements) > 1 else 0.0,
                "min": min(average_improvements),
                "max": max(average_improvements),
            },
        },
        "raw_metrics": {
            "baseline_interventions": total_baseline_interventions,
            "edon_interventions": total_edon_interventions,
            "baseline_stability": total_baseline_stability,
            "edon_stability": total_edon_stability,
        },
    }


def main():
    """Run quick test with 30 episodes (1 batch)."""
    print("="*70)
    print("V4.1 Adaptive Gain System - Quick Test (30 episodes)")
    print("="*70)
    print(f"Configuration: {CONFIG['name']}")
    print(f"  BASE_GAIN: {CONFIG['base_gain']:.2f}")
    print(f"  PREFALL: {CONFIG['prefall_min']:.2f}-{CONFIG['prefall_max']:.2f}")
    print(f"  SAFE threshold: {CONFIG['safe_threshold']:.2f}, gain: {CONFIG['safe_gain']:.2f}")
    print(f"\nTest parameters:")
    print(f"  Episodes: {EPISODES_PER_BATCH}")
    print(f"  Profile: {PROFILE}")
    print(f"  EDON gain: {EDON_GAIN}")
    print(f"  Seed: {SEED}")
    print("="*70)
    
    # Apply configuration
    print(f"\nApplying configuration to run_eval.py...")
    apply_config_to_code(CONFIG)
    print("Done!")
    
    batch_results = []
    
    # Run batches
    for batch_num in range(1, N_BATCHES + 1):
        print(f"\n{'='*70}")
        print(f"Batch {batch_num}/{N_BATCHES}")
        print(f"{'='*70}")
        
        # Run baseline
        baseline_file = run_batch("baseline", batch_num, EPISODES_PER_BATCH)
        if baseline_file is None:
            print(f"  Skipping batch {batch_num} due to baseline error")
            continue
        
        # Run EDON
        edon_file = run_batch("edon", batch_num, EPISODES_PER_BATCH)
        if edon_file is None:
            print(f"  Skipping batch {batch_num} due to EDON error")
            continue
        
        # Compute improvement
        improvement = compute_improvement(baseline_file, edon_file)
        batch_results.append(improvement)
        
        print(f"\n  Batch {batch_num} Results:")
        print(f"    Interventions: {improvement['interventions']:+.1f}%")
        print(f"    Stability: {improvement['stability']:+.1f}%")
        print(f"    Average: {improvement['average']:+.1f}%")
    
    if not batch_results:
        print("\nERROR: No successful batches!")
        return
    
    # Aggregate results
    aggregated = aggregate_results(batch_results)
    
    # Print summary
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    
    if len(batch_results) == 1:
        # Single batch - just show the result
        result = batch_results[0]
        print(f"\nResults ({EPISODES_PER_BATCH} episodes):")
        print(f"  Interventions: {result['interventions']:+.2f}%")
        print(f"  Stability: {result['stability']:+.2f}%")
        print(f"  Average: {result['average']:+.2f}%")
        print(f"\nRaw metrics:")
        print(f"  Baseline interventions: {result['baseline_interventions']:.1f}")
        print(f"  EDON interventions: {result['edon_interventions']:.1f}")
        print(f"  Baseline stability: {result['baseline_stability']:.4f}")
        print(f"  EDON stability: {result['edon_stability']:.4f}")
        
        avg_improvement = result['average']
    else:
        # Multiple batches - show aggregated
        print(f"\nOverall ({aggregated['total_episodes']} episodes):")
        print(f"  Interventions: {aggregated['overall']['interventions']:+.2f}%")
        print(f"  Stability: {aggregated['overall']['stability']:+.2f}%")
        print(f"  Average: {aggregated['overall']['average']:+.2f}%")
        
        print(f"\nPer-batch statistics ({len(batch_results)} batches):")
        print(f"  Interventions: {aggregated['per_batch']['interventions']['mean']:+.2f}% ± {aggregated['per_batch']['interventions']['std']:.2f}%")
        print(f"    Range: {aggregated['per_batch']['interventions']['min']:+.2f}% to {aggregated['per_batch']['interventions']['max']:+.2f}%")
        print(f"  Stability: {aggregated['per_batch']['stability']['mean']:+.2f}% ± {aggregated['per_batch']['stability']['std']:.2f}%")
        print(f"    Range: {aggregated['per_batch']['stability']['min']:+.2f}% to {aggregated['per_batch']['stability']['max']:+.2f}%")
        print(f"  Average: {aggregated['per_batch']['average']['mean']:+.2f}% ± {aggregated['per_batch']['average']['std']:.2f}%")
        print(f"    Range: {aggregated['per_batch']['average']['min']:+.2f}% to {aggregated['per_batch']['average']['max']:+.2f}%")
        
        avg_improvement = aggregated['overall']['average']
    
    # Save results
    results_file = RESULTS_DIR / "quick_test_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "config": CONFIG,
            "aggregated": aggregated,
            "batch_results": batch_results,
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Status
    avg_improvement = aggregated['overall']['average']
    if avg_improvement > 5.0:
        print(f"\n✅ EXCELLENT: {avg_improvement:+.2f}% improvement!")
    elif avg_improvement > 3.0:
        print(f"\n✅ GOOD: {avg_improvement:+.2f}% improvement")
    elif avg_improvement > 0.0:
        print(f"\n⚠️  POSITIVE: {avg_improvement:+.2f}% improvement (below target)")
    else:
        print(f"\n❌ NEGATIVE: {avg_improvement:+.2f}% (needs tuning)")


if __name__ == "__main__":
    main()

