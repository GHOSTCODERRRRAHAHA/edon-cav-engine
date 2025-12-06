"""
Example: Using EDON Robot Stability API

This example shows how to integrate EDON's robot stability control
into a robot control loop.
"""

from edon import EdonClient
import time
import numpy as np

def get_robot_state():
    """Get current robot state (example - replace with your robot's sensors)."""
    # Example: Read from IMU, encoders, etc.
    return {
        "roll": 0.05,  # radians
        "pitch": 0.02,  # radians
        "roll_velocity": 0.1,  # rad/s
        "pitch_velocity": 0.05,  # rad/s
        "com_x": 0.0,  # meters
        "com_y": 0.0,  # meters
    }

def baseline_controller(robot_state):
    """Baseline controller (example - replace with your controller)."""
    # Simple PD controller
    roll = robot_state["roll"]
    pitch = robot_state["pitch"]
    action = np.array([
        -roll * 0.5,  # Correct roll
        -pitch * 0.5,  # Correct pitch
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Other joints
    ])
    return action

def apply_action_to_robot(action):
    """Apply action to robot (example - replace with your robot interface)."""
    print(f"Applying action: {action[:4]}...")  # Print first 4 elements
    # robot.set_joint_torques(action)

def main():
    """Main control loop."""
    # Initialize EDON client
    client = EdonClient(base_url="http://localhost:8002")
    
    # Check if robot stability is available
    health = client.health()
    if not health.get("v8_robot_stability", {}).get("available"):
        print("❌ Robot stability API is not available")
        print("   Make sure v8 models are loaded:")
        print("   - models/edon_v8_strategy_memory_features.pt")
        print("   - models/edon_fail_risk_v1_fixed_v2.pt")
        return
    
    print("✅ Robot stability API is ready")
    print("\nStarting control loop...\n")
    
    # Maintain history for temporal memory
    history = []
    
    # Control loop
    for step in range(100):
        # 1. Get robot state
        robot_state = get_robot_state()
        
        # 2. Get stability control from EDON
        try:
            stability = client.robot_stability(robot_state, history=history[-8:])
        except Exception as e:
            print(f"❌ Error getting stability control: {e}")
            # Fallback to baseline
            action = baseline_controller(robot_state)
            apply_action_to_robot(action)
            continue
        
        # 3. Get baseline action
        baseline_action = baseline_controller(robot_state)
        
        # 4. Apply EDON modulations
        gain_scale = stability['modulations']['gain_scale']
        compliance = stability['modulations']['compliance']
        bias = np.array(stability['modulations']['bias'])
        
        # Scale baseline action
        final_action = baseline_action * gain_scale
        
        # Add bias (scaled by compliance)
        final_action += bias * compliance
        
        # 5. Apply to robot
        apply_action_to_robot(final_action)
        
        # 6. Log status
        if step % 10 == 0:
            print(f"Step {step}:")
            print(f"  Strategy: {stability['strategy_name']}")
            print(f"  Gain Scale: {gain_scale:.2f}")
            print(f"  Compliance: {compliance:.2f}")
            print(f"  Intervention Risk: {stability['intervention_risk']:.3f}")
            print(f"  Latency: {stability['latency_ms']:.1f}ms")
            print()
        
        # 7. Update history (keep last 8 frames)
        history.append(robot_state)
        if len(history) > 8:
            history.pop(0)
        
        # 8. Control frequency (100Hz = 10ms)
        time.sleep(0.01)

if __name__ == "__main__":
    main()

