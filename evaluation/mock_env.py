"""Mock humanoid environment for testing without a real simulator.

This provides a minimal gym-like interface that can be replaced with
your actual humanoid simulation environment.
"""

import numpy as np
from typing import Dict, Any, Optional
import random


class MockHumanoidEnv:
    """
    Realistic mock humanoid environment for testing.
    
    Simulates a humanoid robot with:
    - Balance dynamics (roll/pitch)
    - Center of mass tracking
    - Disturbance response
    - Instability that can lead to falls
    
    Replace this with your actual environment (e.g., gym, mujoco, pybullet).
    The interface should match:
    - reset() -> observation dict
    - step(action) -> (observation, reward, done, info)
    - render() (optional)
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize mock environment."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.seed = seed
        self.step_count = 0
        self.max_steps = 1000
        
        # State (more realistic dynamics)
        self.roll = 0.0
        self.pitch = 0.0
        self.roll_velocity = 0.0
        self.pitch_velocity = 0.0
        self.com_x = 0.0
        self.com_y = 0.0
        self.com_velocity_x = 0.0
        self.com_velocity_y = 0.0
        self.position_x = 0.0
        self.position_y = 0.0
        
        # Physical parameters
        self.mass = 70.0  # kg
        self.inertia = 10.0  # kg*m^2
        self.damping = 0.1  # Damping coefficient
        self.gravity = 9.81
        
        # Disturbance tracking
        self.last_push_time = 0
        self.push_force_x = 0.0
        self.push_force_y = 0.0
        
        # Friction
        self.friction = 0.8
        
        # Goal position
        self.goal_x = 5.0
        self.goal_y = 0.0
    
    def reset(self) -> Dict[str, Any]:
        """Reset environment and return initial observation."""
        self.step_count = 0
        self.roll = np.random.normal(0, 0.05)  # Small initial tilt
        self.pitch = np.random.normal(0, 0.05)
        self.roll_velocity = 0.0
        self.pitch_velocity = 0.0
        self.com_x = 0.0
        self.com_y = 0.0
        self.com_velocity_x = 0.0
        self.com_velocity_y = 0.0
        self.position_x = 0.0
        self.position_y = 0.0
        self.push_force_x = 0.0
        self.push_force_y = 0.0
        self.last_push_time = 0
        
        return self._get_observation()
    
    def step(self, action: Any):
        """
        Step environment with realistic humanoid dynamics.
        
        Args:
            action: Control action array (first 4 elements are balance corrections)
        
        Returns:
            (observation, reward, done, info)
        """
        self.step_count += 1
        dt = 0.01  # 100 Hz control loop
        
        # Parse action (expects array with at least 4 elements)
        if isinstance(action, (list, np.ndarray)) and len(action) >= 4:
            roll_torque = float(action[0])
            pitch_torque = float(action[1])
            com_torque_x = float(action[2])
            com_torque_y = float(action[3])
        else:
            roll_torque = 0.0
            pitch_torque = 0.0
            com_torque_x = 0.0
            com_torque_y = 0.0
        
        # Apply control torques to orientation
        self.roll_velocity += (roll_torque / self.inertia - self.damping * self.roll_velocity) * dt
        self.pitch_velocity += (pitch_torque / self.inertia - self.damping * self.pitch_velocity) * dt
        
        # Update orientation
        self.roll += self.roll_velocity * dt
        self.pitch += self.pitch_velocity * dt
        
        # Gravity effect (tilt increases instability)
        self.roll_velocity += np.sin(self.roll) * self.gravity * 0.1 * dt
        self.pitch_velocity += np.sin(self.pitch) * self.gravity * 0.1 * dt
        
        # Apply external push forces (if any)
        if self.push_force_x != 0 or self.push_force_y != 0:
            self.com_velocity_x += (self.push_force_x / self.mass) * dt
            self.com_velocity_y += (self.push_force_y / self.mass) * dt
            # Decay push forces
            self.push_force_x *= 0.9
            self.push_force_y *= 0.9
            if abs(self.push_force_x) < 0.1:
                self.push_force_x = 0.0
            if abs(self.push_force_y) < 0.1:
                self.push_force_y = 0.0
        
        # Apply COM control
        self.com_velocity_x += (com_torque_x / self.mass - self.damping * self.com_velocity_x) * dt
        self.com_velocity_y += (com_torque_y / self.mass - self.damping * self.com_velocity_y) * dt
        
        # Update COM position
        self.com_x += self.com_velocity_x * dt
        self.com_y += self.com_velocity_y * dt
        
        # Update position (with friction)
        self.position_x += self.com_velocity_x * dt * self.friction
        self.position_y += self.com_velocity_y * dt * self.friction
        
        # Add process noise (sensor/model uncertainty)
        self.roll += np.random.normal(0, 0.002)
        self.pitch += np.random.normal(0, 0.002)
        self.com_x += np.random.normal(0, 0.001)
        self.com_y += np.random.normal(0, 0.001)
        
        # Check if reached goal
        distance_to_goal = np.sqrt(
            (self.position_x - self.goal_x)**2 + 
            (self.position_y - self.goal_y)**2
        )
        success = distance_to_goal < 0.1
        
        # Reward (penalize instability and distance)
        stability_penalty = (abs(self.roll) + abs(self.pitch)) * 0.5
        distance_penalty = distance_to_goal * 0.1
        reward = -stability_penalty - distance_penalty
        if success:
            reward += 10.0
        
        # Done conditions (fall detection)
        fall_threshold = 0.5  # radians (~30 degrees)
        fallen = abs(self.roll) > fall_threshold or abs(self.pitch) > fall_threshold
        
        done = success or fallen or self.step_count >= self.max_steps
        
        # Info
        info = {
            "success": success,
            "fallen": fallen,
            "distance_to_goal": distance_to_goal,
            "step": self.step_count,
            "stability": 1.0 - min(1.0, stability_penalty)
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation with realistic sensor readings."""
        # Compute accelerations from dynamics
        acc_x = -np.sin(self.roll) * self.gravity + self.com_velocity_x * 0.1
        acc_y = -np.sin(self.pitch) * self.gravity + self.com_velocity_y * 0.1
        acc_z = np.cos(self.roll) * np.cos(self.pitch) * self.gravity
        
        return {
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": 0.0,
            "roll_velocity": self.roll_velocity,
            "pitch_velocity": self.pitch_velocity,
            "com_x": self.com_x,
            "com_y": self.com_y,
            "com_velocity_x": self.com_velocity_x,
            "com_velocity_y": self.com_velocity_y,
            "x": self.position_x,
            "y": self.position_y,
            "acc_x": acc_x,
            "acc_y": acc_y,
            "acc_z": acc_z,
            "temp_c": 22.0,
            "humidity": 45.0,
            "aqi": 20,
            "task_id": "navigation",
            "task_complexity": 0.5
        }
    
    def render(self) -> None:
        """Render environment (stub)."""
        pass
    
    def set_friction(self, friction: float) -> None:
        """Set friction coefficient."""
        self.friction = max(0.1, min(2.0, friction))  # Clamp to reasonable range
    
    def apply_external_force(self, force_x: float, force_y: float, force_z: float) -> None:
        """Apply external force (affects COM velocity)."""
        # Apply as impulse to COM velocity
        self.push_force_x = force_x
        self.push_force_y = force_y
        self.last_push_time = self.step_count

