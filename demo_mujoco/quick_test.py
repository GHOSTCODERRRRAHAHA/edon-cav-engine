"""Quick test - just verify it runs."""

print("Starting quick test...")

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("Imports...")
from sim.env import HumanoidEnv
from controllers.baseline_controller import BaselineController
print("✓ Imports successful")

print("\nCreating environment...")
env = HumanoidEnv(dt=0.01, render=False)
print("✓ Environment created")

print("\nCreating controller...")
controller = BaselineController()
print("✓ Controller created")

print("\nResetting environment...")
obs, info = env.reset(seed=42)
print(f"✓ Reset complete. Height: {obs.get('torso_height', 0):.3f}")

print("\nRunning 100 steps...")
for i in range(100):
    action = controller.step(obs)
    obs, done, info = env.step(action)
    if (i + 1) % 20 == 0:
        print(f"  Step {i+1}: height={obs.get('torso_height', 0):.3f}, roll={obs.get('roll', 0):.3f}, done={done}")

print("\n✓ Test complete! Environment is working.")
print(f"Final state: height={obs.get('torso_height', 0):.3f}, roll={obs.get('roll', 0):.3f}")

