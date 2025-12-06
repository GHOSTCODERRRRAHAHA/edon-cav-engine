import json
import numpy as np

def load_stats(prefix):
    interventions = []
    stability = []
    for s in range(5):
        path = f"results/{prefix}_highstress_seed_{s}.json"
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            # Try different possible paths for the metrics
            if "summary" in d:
                interventions.append(d["summary"]["interventions_per_episode"])
                stability.append(d["summary"]["avg_stability"])
            elif "run_metrics" in d:
                interventions.append(d["run_metrics"]["interventions_per_episode"])
                stability.append(d["run_metrics"]["stability_avg"])
            else:
                interventions.append(d.get("interventions_per_episode", 0))
                stability.append(d.get("stability_avg", 0))
        except FileNotFoundError:
            print(f"Warning: {path} not found, skipping seed {s}")
            continue
    if len(interventions) == 0:
        return None, None
    return np.mean(interventions), np.mean(stability)

base_int, base_stab = load_stats("baseline")
edon_int, edon_stab = load_stats("edon_v6_prefall_weighted")

if base_int is None or edon_int is None:
    print("Error: Could not load all required files")
    exit(1)

print("Baseline:", base_int, base_stab)
print("EDON v6 weighted:", edon_int, edon_stab)

delta_int = 100.0 * (base_int - edon_int) / base_int
delta_stab = 100.0 * (edon_stab - base_stab) / base_stab
avg = 0.5 * (delta_int + delta_stab)

print("dInterventions%:", delta_int)
print("dStability%:", delta_stab)
print("Average%:", avg)

