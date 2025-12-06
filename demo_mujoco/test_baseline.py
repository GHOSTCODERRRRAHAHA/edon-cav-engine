"""Quick test to verify baseline controller and environment work."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sim.env import HumanoidEnv
from controllers.baseline_controller import BaselineController
import numpy as np

print("Testing MuJoCo environment and baseline controller...")
print("="*60)

try:
    # Create environment
    print("1. Creating environment...")
    env = HumanoidEnv(dt=0.01, render=False)
    print("   ✓ Environment created")
    
    # Create controller
    print("2. Creating baseline controller...")
    controller = BaselineController()
    print("   ✓ Controller created")
    
    # Reset environment
    print("3. Resetting environment...")
    obs, info = env.reset(seed=42)
    print(f"   ✓ Environment reset")
    print(f"   Observation keys: {list(obs.keys())}")
    print(f"   Initial state: roll={obs.get('roll', 0):.3f}, pitch={obs.get('pitch', 0):.3f}, height={obs.get('torso_height', 0):.3f}")
    
    # Run a few steps
    print("4. Running 10 steps...")
    for i in range(10):
        action = controller.step(obs)
        print(f"   Step {i+1}: action shape={action.shape}, action[0:3]={action[:3]}")
        obs, done, info = env.step(action)
        print(f"   Step {i+1}: roll={obs.get('roll', 0):.3f}, pitch={obs.get('pitch', 0):.3f}, height={obs.get('torso_height', 0):.3f}")
        if done:
            print(f"   Episode ended at step {i+1}")
            break
    
    print("\n" + "="*60)
    print("✓ Test passed! Environment and controller are working.")
    print("="*60)
    
except Exception as e:
    print(f"\n✗ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

