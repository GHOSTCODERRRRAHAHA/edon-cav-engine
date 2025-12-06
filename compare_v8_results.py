"""Compare v8 memory+features results with baseline"""
import json
from pathlib import Path

# Load baseline
baseline_path = Path("results/baseline_high_stress_v44.json")
if not baseline_path.exists():
    baseline_path = Path("results/baseline_high_stress.json")

if baseline_path.exists():
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)
    baseline_int = baseline.get("interventions_per_episode", 0)
    baseline_stab = baseline.get("stability_avg", 0)
    baseline_len = baseline.get("avg_episode_length", 0)
else:
    # Use typical baseline values
    baseline_int = 40.2
    baseline_stab = 0.0206
    baseline_len = 310.0
    print(f"[COMPARE] Baseline file not found, using typical values")

# Load EDON v8 results
edon_path = Path("results/edon_v8_memory_features.json")
with open(edon_path, 'r') as f:
    edon = json.load(f)

# Extract metrics - check multiple possible locations
edon_int = 0
edon_stab = 0
edon_len = 0

if "run_metrics" in edon:
    edon_int = edon["run_metrics"].get("interventions_per_episode", 0)
    edon_stab = edon["run_metrics"].get("stability_avg", 0)
    edon_len = edon["run_metrics"].get("avg_episode_length", 0)
elif "summary" in edon:
    edon_int = edon["summary"].get("interventions_per_episode", 0)
    edon_stab = edon["summary"].get("stability_avg", 0)
    edon_len = edon["summary"].get("avg_episode_length", 0)
else:
    edon_int = edon.get("interventions_per_episode", 0)
    edon_stab = edon.get("stability_avg", 0)
    edon_len = edon.get("avg_episode_length", 0)

# If still zero, try to compute from episodes
if edon_int == 0 and "episodes" in edon:
    interventions = [ep.get("interventions", 0) for ep in edon["episodes"]]
    if interventions:
        edon_int = sum(interventions) / len(interventions)

# Compute deltas
delta_int = 100 * (baseline_int - edon_int) / baseline_int if baseline_int > 0 else 0
delta_stab = 100 * (baseline_stab - edon_stab) / baseline_stab if baseline_stab > 0 else 0
delta_len = 100 * (edon_len - baseline_len) / baseline_len if baseline_len > 0 else 0

# Print comparison
print("=" * 70)
print("EDON v8 Memory+Features vs Baseline Comparison")
print("=" * 70)
print(f"\nBaseline (from {baseline_path.name if baseline_path.exists() else 'typical values'}):")
print(f"  Interventions/episode: {baseline_int:.2f}")
print(f"  Stability (avg): {baseline_stab:.4f}")
print(f"  Episode length (avg): {baseline_len:.1f} steps")

print(f"\nEDON v8 Memory+Features:")
print(f"  Interventions/episode: {edon_int:.2f}")
print(f"  Stability (avg): {edon_stab:.4f}")
print(f"  Episode length (avg): {edon_len:.1f} steps")

print(f"\n" + "=" * 70)
print("DELTA (Improvement)")
print("=" * 70)
print(f"\nΔInterventions: {delta_int:+.2f}%")
if delta_int >= 10:
    print(f"  ✅ GOAL MET: ≥10% reduction")
elif delta_int > 0:
    print(f"  ⚠️  PARTIAL: {delta_int:.2f}% reduction (need ≥10%)")
else:
    print(f"  ❌ REGRESSION: {abs(delta_int):.2f}% increase")

print(f"\nΔStability: {delta_stab:+.2f}%")
if abs(delta_stab) <= 5:
    print(f"  ✅ GOAL MET: Within ±5%")
else:
    print(f"  ❌ OUT OF RANGE: {abs(delta_stab):.2f}% change (target: ±5%)")

print(f"\nΔEpisode Length: {delta_len:+.2f}%")

print(f"\n" + "=" * 70)
print("OVERALL ASSESSMENT")
print("=" * 70)
if delta_int >= 10 and abs(delta_stab) <= 5:
    print("✅ GOAL ACHIEVED: ≥10% intervention reduction with stable stability!")
elif delta_int > 0 and abs(delta_stab) <= 5:
    print(f"⚠️  PARTIAL SUCCESS: {delta_int:.2f}% intervention reduction (need ≥10%)")
    print("   Stability constraint met (±5%)")
else:
    print("❌ GOAL NOT MET:")
    if delta_int < 10:
        print(f"   - Interventions: {delta_int:+.2f}% (need ≥10%)")
    if abs(delta_stab) > 5:
        print(f"   - Stability: {delta_stab:+.2f}% (need ±5%)")

print("=" * 70)

