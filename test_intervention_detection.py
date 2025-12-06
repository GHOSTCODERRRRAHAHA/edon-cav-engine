"""Test intervention detection with a simple episode"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from run_eval import make_humanoid_env
from metrics.edon_v8_metrics import compute_episode_metrics_v8

print("="*70)
print("TESTING INTERVENTION DETECTION")
print("="*70)

# Create environment
env = make_humanoid_env(seed=42, profile="high_stress")
obs = env.reset()

# Collect episode with known bad actions (should cause interventions)
episode_data = []
for i in range(50):
    # Use a very bad action that should cause falls
    bad_action = np.ones(10) * 5.0  # Large action
    obs, reward, done, info = env.step(bad_action)
    
    # Store step data in the format expected by compute_episode_metrics_v8
    episode_data.append({
        "obs": obs,
        "info": info,
        "done": done
    })
    
    # Print info dict structure
    if i < 5:
        print(f"\nStep {i} info dict:")
        print(f"  Type: {type(info)}")
        if isinstance(info, dict):
            print(f"  Keys: {list(info.keys())}")
            print(f"  intervention: {info.get('intervention', 'NOT SET')}")
            print(f"  fallen: {info.get('fallen', 'NOT SET')}")
        else:
            print(f"  Value: {info}")
    
    if done:
        obs = env.reset()

# Compute metrics
print("\n" + "="*70)
print("COMPUTING METRICS")
print("="*70)

metrics = compute_episode_metrics_v8(episode_data)
print(f"\nDetected interventions: {metrics.get('interventions', 0)}")
print(f"Episode length: {metrics.get('episode_length', 0)}")

# Manual count
manual_count = 0
for i, step in enumerate(episode_data):
    info = step.get("info", {})
    if isinstance(info, dict):
        if info.get("intervention", False) or info.get("fallen", False):
            manual_count += 1
            if manual_count <= 5:
                print(f"  Step {i}: intervention={info.get('intervention', False)}, fallen={info.get('fallen', False)}")

print(f"\nManual count: {manual_count}")

if metrics.get('interventions', 0) == 0 and manual_count > 0:
    print("\n❌ BUG: Metrics function didn't detect interventions that exist!")
elif metrics.get('interventions', 0) > 0:
    print("\n✅ Interventions are being detected correctly")
else:
    print("\n⚠️  No interventions detected - either detection is broken or actions weren't bad enough")

print("="*70)

