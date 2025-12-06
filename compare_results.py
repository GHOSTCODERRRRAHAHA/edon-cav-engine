import json

# Load both results
with open('eval_output.json', 'r') as f:
    edon_data = json.load(f)

with open('baseline_output.json', 'r') as f:
    baseline_data = json.load(f)

edon_ep = edon_data['episodes'][0]
baseline_ep = baseline_data['episodes'][0]

print("="*70)
print("EDON v7 vs BASELINE Comparison")
print("="*70)
print(f"\nProfile: {edon_data['profile']}")
print(f"Seed: {edon_data['seed']}")
print(f"Episodes: 1")

print("\n" + "-"*70)
print("KEY METRICS")
print("-"*70)
print(f"Interventions:     EDON={edon_ep['interventions']:3d}, Baseline={baseline_ep['interventions']:3d}, Diff={edon_ep['interventions']-baseline_ep['interventions']:+3d}")
print(f"Stability Score:   EDON={edon_ep['stability_score']:.6f}, Baseline={baseline_ep['stability_score']:.6f}, Diff={edon_ep['stability_score']-baseline_ep['stability_score']:+.6f}")
print(f"Episode Length:    EDON={edon_ep['episode_length']:4d}, Baseline={baseline_ep['episode_length']:4d}, Diff={edon_ep['episode_length']-baseline_ep['episode_length']:+4d}")
print(f"Episode Time:      EDON={edon_ep['episode_time']:.3f}s, Baseline={baseline_ep['episode_time']:.3f}s, Diff={edon_ep['episode_time']-baseline_ep['episode_time']:+.3f}s")
print(f"Success:           EDON={edon_ep['success']}, Baseline={baseline_ep['success']}")

print("\n" + "-"*70)
print("STABILITY METRICS")
print("-"*70)
print(f"Roll RMS:         EDON={edon_ep['roll_rms']:.6f}, Baseline={baseline_ep['roll_rms']:.6f}, Diff={edon_ep['roll_rms']-baseline_ep['roll_rms']:+.6f}")
print(f"Pitch RMS:        EDON={edon_ep['pitch_rms']:.6f}, Baseline={baseline_ep['pitch_rms']:.6f}, Diff={edon_ep['pitch_rms']-baseline_ep['pitch_rms']:+.6f}")
print(f"Roll Max:         EDON={edon_ep['roll_max']:.6f}, Baseline={baseline_ep['roll_max']:.6f}, Diff={edon_ep['roll_max']-baseline_ep['roll_max']:+.6f}")
print(f"Pitch Max:        EDON={edon_ep['pitch_max']:.6f}, Baseline={baseline_ep['pitch_max']:.6f}, Diff={edon_ep['pitch_max']-baseline_ep['pitch_max']:+.6f}")
print(f"COM Deviation:    EDON={edon_ep['com_deviation']:.6f}, Baseline={baseline_ep['com_deviation']:.6f}, Diff={edon_ep['com_deviation']-baseline_ep['com_deviation']:+.6f}")

print("\n" + "-"*70)
print("EVENT COUNTS")
print("-"*70)
print(f"Prefall Events:   EDON={edon_ep['prefall_events']:3d}, Baseline={baseline_ep['prefall_events']:3d}, Diff={edon_ep['prefall_events']-baseline_ep['prefall_events']:+3d}")
print(f"Fail Events:       EDON={edon_ep['fail_events']:3d}, Baseline={baseline_ep['fail_events']:3d}, Diff={edon_ep['fail_events']-baseline_ep['fail_events']:+3d}")
print(f"Freeze Events:     EDON={edon_ep['freeze_events']:3d}, Baseline={baseline_ep['freeze_events']:3d}, Diff={edon_ep['freeze_events']-baseline_ep['freeze_events']:+3d}")

print("\n" + "-"*70)
print("INTERVENTION TIMING")
print("-"*70)
edon_first = edon_ep['intervention_times'][0] if edon_ep['intervention_times'] else None
edon_last = edon_ep['intervention_times'][-1] if edon_ep['intervention_times'] else None
baseline_first = baseline_ep['intervention_times'][0] if baseline_ep['intervention_times'] else None
baseline_last = baseline_ep['intervention_times'][-1] if baseline_ep['intervention_times'] else None

print(f"First Intervention: EDON={edon_first:.3f}s, Baseline={baseline_first:.3f}s")
print(f"Last Intervention:  EDON={edon_last:.3f}s, Baseline={baseline_last:.3f}s")
print(f"Intervention Span:   EDON={edon_last-edon_first:.3f}s, Baseline={baseline_last-baseline_first:.3f}s")

print("\n" + "-"*70)
print("EDON EPISODE SCORE")
print("-"*70)
edon_score = edon_data.get('run_metrics', {}).get('edon_score', 'N/A')
baseline_score = baseline_data.get('run_metrics', {}).get('edon_score', 'N/A')
print(f"EDON Score:        EDON={edon_score}, Baseline={baseline_score}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
if edon_ep['interventions'] < baseline_ep['interventions']:
    print("[+] EDON v7 has FEWER interventions (better)")
elif edon_ep['interventions'] > baseline_ep['interventions']:
    print("[-] EDON v7 has MORE interventions (worse)")
else:
    print("[=] Same number of interventions")

if edon_ep['stability_score'] < baseline_ep['stability_score']:
    print("[+] EDON v7 has BETTER stability (lower is better)")
elif edon_ep['stability_score'] > baseline_ep['stability_score']:
    print("[-] EDON v7 has WORSE stability (higher is worse)")
else:
    print("[=] Same stability score")

if edon_ep['episode_time'] > baseline_ep['episode_time'] * 10:
    print(f"[!] EDON v7 is {edon_ep['episode_time']/baseline_ep['episode_time']:.1f}x SLOWER (due to API calls)")
print("="*70)
