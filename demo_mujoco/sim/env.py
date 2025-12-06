"""MuJoCo environment wrapper for humanoid robot."""

import mujoco
import mujoco.viewer
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import json
import os


class HumanoidEnv:
    """
    MuJoCo environment wrapper for humanoid robot.
    
    Provides a clean interface for:
    - Resetting with seed and disturbance script
    - Stepping with actions
    - Getting state observations
    - Applying disturbances
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        dt: float = 0.01,
        render: bool = False,
        sensor_noise_std: float = 0.0,
        actuator_delay_steps: tuple = (0, 0),
        friction_min: float = 0.5,
        friction_max: float = 1.0,
        fatigue_enabled: bool = False,
        fatigue_degradation: float = 0.0,
        floor_incline_range: tuple = (0.0, 0.0),
        height_variation_range: tuple = (0.0, 0.0)
    ):
        """
        Initialize MuJoCo environment.
        
        Args:
            model_path: Path to MuJoCo XML model file. If None, uses built-in humanoid.
            dt: Simulation timestep (default: 0.01 = 100Hz)
            render: Whether to enable rendering
        """
        self.dt = dt
        self.render = render
        self.viewer = None
        
        # Store realism parameters
        self.sensor_noise_std = sensor_noise_std
        self.actuator_delay_steps = actuator_delay_steps
        self.friction_min = friction_min
        self.friction_max = friction_max
        self.fatigue_enabled = fatigue_enabled
        self.fatigue_degradation = fatigue_degradation
        self.floor_incline_range = floor_incline_range
        self.height_variation_range = height_variation_range
        
        # Load model
        if model_path is None:
            # Use MuJoCo's built-in humanoid model
            model_path = self._get_builtin_humanoid()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Find key body IDs
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        if self.torso_id < 0:
            # Try alternative names
            for name in ["pelvis", "root", "base"]:
                self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if self.torso_id >= 0:
                    break
        
        if self.torso_id < 0:
            # Default to body 0 (root)
            self.torso_id = 0
        
        # Get joint IDs for control
        self.actuator_ids = list(range(self.model.nu))
        
        # State tracking
        self.step_count = 0
        self.max_steps = 100000  # Increased to allow longer episodes
        self.disturbance_script: List[Dict[str, Any]] = []
        self.disturbance_index = 0
        self._pending_force_reset = False
        
        # Terrain (heightfield)
        self.terrain = None
        self.terrain_height = 0.0
        
        # Latency jitter
        self.latency_jitter = 0.0
        self.latency_jitter_range = (0.0, 0.05)  # 0-50ms jitter
        
        # Load shift
        self.load_mass = 0.0
        self.load_position = np.array([0.0, 0.0, 0.0])
        
        # Smooth disturbance tracking (for gradual force application)
        self.active_push_force = np.array([0.0, 0.0, 0.0])
        self.push_decay_rate = 0.9  # Decay per step (matches original)
        self.push_start_time = None
        
        # Realism features (matching training/eval environment)
        self.sensor_noise_std = 0.0  # Will be set by stress profile
        self.actuator_delay_steps = (0, 0)  # Will be set by stress profile
        self.actuator_delay_buffer = []  # Buffer for delayed actions
        self.friction_min = 0.5
        self.friction_max = 1.0
        self.current_friction = 1.0
        self.fatigue_enabled = False
        self.fatigue_degradation = 0.0
        self.floor_incline_range = (0.0, 0.0)
        self.current_floor_incline = 0.0
        self.height_variation_range = (0.0, 0.0)
        self.current_height_variation = 0.0
        
        # Find floor geom for friction/incline modification
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if self.floor_geom_id < 0:
            # Try to find any plane geom
            for i in range(self.model.ngeom):
                geom_type = self.model.geom_type[i]
                if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
                    self.floor_geom_id = i
                    break
        
        # Initialize viewer if rendering
        if self.render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
    
    def _get_builtin_humanoid(self) -> str:
        """Get path to built-in humanoid model or create a simple one."""
        # Use the simple humanoid XML file
        xml_path = os.path.join(os.path.dirname(__file__), "simple_humanoid.xml")
        if os.path.exists(xml_path):
            return xml_path
        return self._create_simple_humanoid()
    
    def _create_simple_humanoid(self) -> str:
        """Create a simple humanoid model XML."""
        xml_path = os.path.join(os.path.dirname(__file__), "simple_humanoid.xml")
        if os.path.exists(xml_path):
            return xml_path
        
        # Create simple humanoid XML
        xml_content = """<?xml version="1.0"?>
<mujoco model="simple_humanoid">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0.2"/>
    <material name="body" rgba="0.7 0.7 0.8 1"/>
  </asset>
  
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="10 10 0.1" material="grid" condim="3" friction="1 0.005 0.0001"/>
    
    <!-- Torso -->
    <body name="torso" pos="0 0 1.0">
      <joint name="root_x" type="slide" axis="1 0 0" damping="10"/>
      <joint name="root_y" type="slide" axis="0 1 0" damping="10"/>
      <joint name="root_z" type="slide" axis="0 0 1" damping="10"/>
      <joint name="root_rot_x" type="hinge" axis="1 0 0" damping="1"/>
      <joint name="root_rot_y" type="hinge" axis="0 1 0" damping="1"/>
      <joint name="root_rot_z" type="hinge" axis="0 0 1" damping="1"/>
      <geom name="torso_geom" type="box" size="0.15 0.1 0.2" mass="10" material="body"/>
      <inertial pos="0 0 0" mass="10" diaginertia="0.5 0.5 0.5"/>
      
      <!-- Left leg -->
      <body name="left_thigh" pos="0.05 -0.05 -0.2">
        <joint name="left_hip_x" type="hinge" axis="1 0 0" range="-1.57 1.57" damping="1"/>
        <joint name="left_hip_y" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="1"/>
        <geom name="left_thigh_geom" type="capsule" size="0.05" fromto="0 0 0 0 0 -0.3" mass="3" material="body"/>
        <body name="left_shank" pos="0 0 -0.3">
          <joint name="left_knee" type="hinge" axis="0 1 0" range="-2.0 0" damping="1"/>
          <geom name="left_shank_geom" type="capsule" size="0.04" fromto="0 0 0 0 0 -0.3" mass="2" material="body"/>
          <body name="left_foot" pos="0 0 -0.3">
            <geom name="left_foot_geom" type="box" size="0.08 0.04 0.02" mass="1" material="body"/>
          </body>
        </body>
      </body>
      
      <!-- Right leg -->
      <body name="right_thigh" pos="0.05 0.05 -0.2">
        <joint name="right_hip_x" type="hinge" axis="1 0 0" range="-1.57 1.57" damping="1"/>
        <joint name="right_hip_y" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="1"/>
        <geom name="right_thigh_geom" type="capsule" size="0.05" fromto="0 0 0 0 0 -0.3" mass="3" material="body"/>
        <body name="right_shank" pos="0 0 -0.3">
          <joint name="right_knee" type="hinge" axis="0 1 0" range="-2.0 0" damping="1"/>
          <geom name="right_shank_geom" type="capsule" size="0.04" fromto="0 0 0 0 0 -0.3" mass="2" material="body"/>
          <body name="right_foot" pos="0 0 -0.3">
            <geom name="right_foot_geom" type="box" size="0.08 0.04 0.02" mass="1" material="body"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <motor name="root_x" joint="root_x" gear="100"/>
    <motor name="root_y" joint="root_y" gear="100"/>
    <motor name="root_z" joint="root_z" gear="100"/>
    <motor name="root_rot_x" joint="root_rot_x" gear="50"/>
    <motor name="root_rot_y" joint="root_rot_y" gear="50"/>
    <motor name="root_rot_z" joint="root_rot_z" gear="50"/>
    <motor name="left_hip_x" joint="left_hip_x" gear="100"/>
    <motor name="left_hip_y" joint="left_hip_y" gear="100"/>
    <motor name="left_knee" joint="left_knee" gear="100"/>
    <motor name="right_hip_x" joint="right_hip_x" gear="100"/>
    <motor name="right_hip_y" joint="right_hip_y" gear="100"/>
    <motor name="right_knee" joint="right_knee" gear="100"/>
  </actuator>
</mujoco>
"""
        with open(xml_path, 'w') as f:
            f.write(xml_content)
        return xml_path
    
    def reset(
        self,
        seed: Optional[int] = None,
        disturbance_script: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset environment.
        
        Args:
            seed: Random seed for reproducibility
            disturbance_script: List of disturbance events to replay
        
        Returns:
            (observation, info) tuple
        """
        if seed is not None:
            np.random.seed(seed)
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
        
        self.step_count = 0
        self.disturbance_script = disturbance_script or []
        self.disturbance_index = 0
        
        # Reset intervention logging for new episode
        if hasattr(self, '_intervention_count'):
            delattr(self, '_intervention_count')
        
        # Reset terrain
        self.terrain = None
        self.terrain_height = 0.0
        
        # Reset load
        self.load_mass = 0.0
        self.load_position = np.array([0.0, 0.0, 0.0])
        
        # Reset latency jitter
        self.latency_jitter = 0.0
        
        # Reset smooth disturbance tracking
        self.active_push_force = np.array([0.0, 0.0, 0.0])
        self.push_start_time = None
        
        # Reset realism features
        self.actuator_delay_buffer = []
        # Randomize friction
        if self.friction_min < self.friction_max:
            self.current_friction = np.random.uniform(self.friction_min, self.friction_max)
            if self.floor_geom_id >= 0:
                # Update floor friction in model
                self.model.geom_friction[self.floor_geom_id, 0] = self.current_friction
        else:
            self.current_friction = 1.0
        
        # Randomize floor incline
        if self.floor_incline_range[0] != self.floor_incline_range[1]:
            self.current_floor_incline = np.random.uniform(
                self.floor_incline_range[0], 
                self.floor_incline_range[1]
            )
            # Apply incline by rotating the floor (simplified - would need to modify geom)
            # For now, we'll apply it as a gravity component
        else:
            self.current_floor_incline = 0.0
        
        # Randomize height variation
        if self.height_variation_range[0] != self.height_variation_range[1]:
            self.current_height_variation = np.random.uniform(
                self.height_variation_range[0],
                self.height_variation_range[1]
            )
        else:
            self.current_height_variation = 0.0
        
        obs = self._get_observation()
        info = {"step": 0, "reset": True}
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        """
        Step environment.
        
        Args:
            action: Control action (torques for actuators)
        
        Returns:
            (observation, done, info)
        """
        # Don't increment here - increment after step to match MuJoCo's internal step count
        
        # Apply actuator delay (real motors have latency)
        if self.actuator_delay_steps[1] > 0:
            # Add current action to buffer
            self.actuator_delay_buffer.append(action.copy())
            # Keep buffer size limited to max delay
            max_buffer_size = self.actuator_delay_steps[1] + 1
            if len(self.actuator_delay_buffer) > max_buffer_size:
                self.actuator_delay_buffer.pop(0)
            # Determine delay (random within range)
            delay = np.random.randint(self.actuator_delay_steps[0], self.actuator_delay_steps[1] + 1)
            # Use delayed action if available
            if len(self.actuator_delay_buffer) > delay:
                action = self.actuator_delay_buffer[-(delay+1)]  # Get action from delay steps ago
            else:
                # Not enough history, use oldest available (or current if buffer empty)
                if len(self.actuator_delay_buffer) > 0:
                    action = self.actuator_delay_buffer[0]  # Use oldest
                # Otherwise use current action (already in buffer)
        
        # Apply fatigue (degrade actuator performance over time)
        if self.fatigue_enabled and self.fatigue_degradation > 0:
            # Calculate fatigue factor (linear degradation over episode)
            max_steps = 1000  # Assume 10s episode
            fatigue_factor = 1.0 - (self.fatigue_degradation * (self.step_count / max_steps))
            fatigue_factor = max(0.5, fatigue_factor)  # Don't degrade below 50%
            action = action * fatigue_factor
        
        # Apply action with aggressive clamping to prevent instability
        # Much tighter limits to prevent numerical explosion
        action_clamped = np.clip(action[:self.model.nu], -20.0, 20.0)
        
        # Additional safety: check for NaN/Inf in action
        if not np.all(np.isfinite(action_clamped)):
            if not hasattr(self, '_warned_action'):
                print(f"  [Env] Warning: Invalid action detected at step {self.step_count}, zeroing...")
                self._warned_action = True
            action_clamped = np.zeros_like(action_clamped)
        
        # Clamp joint velocities before applying action to prevent QACC explosion
        # If velocities are too high, the physics solver can't converge
        max_vel = 10.0  # Reasonable max velocity
        for i in range(min(len(self.data.qvel), self.model.nv)):
            if abs(self.data.qvel[i]) > max_vel:
                self.data.qvel[i] = np.clip(self.data.qvel[i], -max_vel, max_vel)
        
        self.data.ctrl[:] = action_clamped
        
        # Apply disturbances
        self._apply_disturbances()
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Safety check: detect instability immediately after step
        # Check for NaN/Inf in state and accelerations
        qacc_invalid = not np.all(np.isfinite(self.data.qacc))
        qpos_invalid = not np.all(np.isfinite(self.data.qpos))
        qvel_invalid = not np.all(np.isfinite(self.data.qvel))
        
        if qacc_invalid or qpos_invalid or qvel_invalid:
            # Find which DOF has the issue
            if qacc_invalid:
                invalid_dofs = np.where(~np.isfinite(self.data.qacc))[0]
                # Don't print every time - only first occurrence
                if not hasattr(self, '_warned_qacc'):
                    print(f"  [Env] QACC instability at DOF {invalid_dofs[0]} (step {self.step_count}) - clamping state")
                    self._warned_qacc = True
            
            # Try to recover by clamping state
            # Clamp positions to reasonable bounds
            self.data.qpos[:] = np.clip(self.data.qpos, -10.0, 10.0)
            # Clamp velocities to prevent explosion
            self.data.qvel[:] = np.clip(self.data.qvel, -50.0, 50.0)
            # Zero out invalid accelerations
            self.data.qacc[:] = np.where(np.isfinite(self.data.qacc), self.data.qacc, 0.0)
            # Forward kinematics to recompute
            mujoco.mj_forward(self.model, self.data)
        
        # Note: Forces are now handled in _apply_disturbances with smooth decay
        # No need to reset here - forces decay gradually
        
        # Update viewer
        if self.viewer is not None:
            self.viewer.sync()
        
        # Increment step count
        self.step_count += 1
        
        # Get observation first (before checking done)
        obs = self._get_observation()
        
        # Check termination (but don't end early - let episode run full duration)
        # Only end on max_steps or catastrophic failure (NaN/Inf)
        done = False
        if self.step_count >= self.max_steps:
            done = True
        else:
            # Check for NaN/Inf (catastrophic failure that would crash)
            if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
                done = True
                print(f"  [Env] Catastrophic failure detected at step {self.step_count} (NaN/Inf)")
        
        # Intervention detection (matches original training/eval environment)
        # Intervention is triggered when tilt exceeds 0.35 rad (~20 degrees)
        # This matches the original FAIL_LIMIT threshold
        roll_abs = abs(obs.get('roll', 0.0))
        pitch_abs = abs(obs.get('pitch', 0.0))
        intervention_threshold = 0.35  # radians (~20 degrees) - matches original FAIL_LIMIT
        intervention_detected = (roll_abs > intervention_threshold) or (pitch_abs > intervention_threshold)
        
        # Diagnostic: Log intervention details (first few only, and only if detected)
        if intervention_detected and not hasattr(self, '_intervention_logged'):
            # Track that we've logged for this episode (reset in reset())
            if not hasattr(self, '_intervention_count'):
                self._intervention_count = 0
            self._intervention_count += 1
            if self._intervention_count <= 3:  # Log first 3 interventions
                print(f"  [Env] Intervention #{self._intervention_count} at step {self.step_count}: "
                      f"roll={roll_abs:.3f} rad ({roll_abs*57.3:.1f}°), "
                      f"pitch={pitch_abs:.3f} rad ({pitch_abs*57.3:.1f}°), "
                      f"threshold=0.35 rad (20°)")
        
        # Info dict
        info = {
            "step": self.step_count,
            "disturbance_active": self.disturbance_index < len(self.disturbance_script),
            "fall_detected": obs.get('torso_height', 1.0) < 0.3,  # Inform but don't stop
            "intervention_detected": intervention_detected,  # Matches original environment
            "intervention": intervention_detected,  # Also set for compatibility
            "fallen": intervention_detected,  # Also set for compatibility with original metrics
            # Diagnostic info
            "roll_abs": roll_abs,
            "pitch_abs": pitch_abs,
            "intervention_threshold": intervention_threshold,
        }
        
        return obs, done, info
    
    def _apply_disturbances(self):
        """Apply disturbances from script with smooth gradual forces."""
        current_time = self.step_count * self.dt
        
        # Decay existing push force (smooth gradual application, like original)
        if np.linalg.norm(self.active_push_force) > 0.1:
            self.active_push_force *= self.push_decay_rate
            # Apply decaying force
            self.data.xfrc_applied[self.torso_id, :3] = self.active_push_force
        else:
            # Force has decayed, clear it
            self.active_push_force = np.array([0.0, 0.0, 0.0])
            self.data.xfrc_applied[self.torso_id, :3] = 0.0
        
        # Check for new disturbances at current time
        while (self.disturbance_index < len(self.disturbance_script) and
               self.disturbance_script[self.disturbance_index].get("time", 0) <= current_time):
            dist = self.disturbance_script[self.disturbance_index]
            dist_type = dist.get("type")
            
            if dist_type == "push":
                # Start new push force (will decay gradually over time)
                force = np.array(dist.get("force", [0, 0, 0]))
                # Initialize active push force (will decay each step)
                self.active_push_force = force.copy()
                self.push_start_time = current_time
                # Apply initial force
                self.data.xfrc_applied[self.torso_id, :3] = self.active_push_force
            
            elif dist_type == "terrain":
                # Apply terrain height (simplified - would need heightfield geom)
                self.terrain_height = dist.get("height", 0.0)
            
            elif dist_type == "load_shift":
                # Shift load position
                self.load_position = np.array(dist.get("position", [0, 0, 0]))
            
            elif dist_type == "latency_jitter":
                # Set latency jitter
                self.latency_jitter = dist.get("jitter", 0.0)
            
            self.disturbance_index += 1
    
    def _check_done(self) -> bool:
        """Check if episode is done."""
        # Don't end early - let the episode run for full duration
        # Only check for catastrophic failures that would crash the simulation
        if self.step_count >= self.max_steps:
            return True
        
        # Check for NaN/Inf in state (this would crash)
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            return True
        
        # Don't end on falls - let the episode continue
        # The metrics tracker will count falls, but we keep running
        return False
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        # Get torso orientation (quaternion)
        torso_quat = self.data.xquat[self.torso_id]
        
        # Convert quaternion to roll/pitch/yaw
        # MuJoCo quaternion format: [w, x, y, z]
        w, x, y, z = torso_quat
        
        # Roll (rotation around X-axis)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (rotation around Y-axis)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Clamp angles to reasonable range
        roll = np.clip(roll, -np.pi/2, np.pi/2)
        pitch = np.clip(pitch, -np.pi/2, np.pi/2)
        
        # Get angular velocities from root rotation joints
        # Assuming qvel structure: [root_x, root_y, root_z, root_rot_x, root_rot_y, root_rot_z, ...]
        if self.data.qvel.size >= 6:
            roll_velocity = np.clip(self.data.qvel[3], -10.0, 10.0)  # root_rot_x velocity
            pitch_velocity = np.clip(self.data.qvel[4], -10.0, 10.0)  # root_rot_y velocity
        else:
            roll_velocity = 0.0
            pitch_velocity = 0.0
        
        # Get center of mass
        com = self.data.subtree_com[self.torso_id]
        com_x = np.clip(com[0], -1.0, 1.0)
        com_y = np.clip(com[1], -1.0, 1.0)
        
        # Get joint positions and velocities
        if self.data.qpos.size > 6:
            joint_pos = self.data.qpos[6:].copy()  # Skip root joints
        else:
            joint_pos = np.zeros(6)
        
        if self.data.qvel.size > 6:
            joint_vel = self.data.qvel[6:].copy()
        else:
            joint_vel = np.zeros(6)
        
        # Get IMU-like data (torso linear acceleration)
        torso_linacc = np.clip(self.data.qacc[:3], -100.0, 100.0).copy()
        
        # Get torso height and clamp to reasonable range
        torso_height = float(self.data.xpos[self.torso_id][2])
        torso_height = np.clip(torso_height, 0.0, 5.0)  # Clamp to 0-5m
        
        # Check for NaN/Inf values
        if not np.isfinite(roll) or not np.isfinite(pitch):
            roll = 0.0
            pitch = 0.0
        
        # Apply sensor noise (real IMU and encoders have noise)
        if self.sensor_noise_std > 0:
            roll += np.random.normal(0, self.sensor_noise_std)
            pitch += np.random.normal(0, self.sensor_noise_std)
            roll_velocity += np.random.normal(0, self.sensor_noise_std * 0.5)
            pitch_velocity += np.random.normal(0, self.sensor_noise_std * 0.5)
            com_x += np.random.normal(0, self.sensor_noise_std * 0.1)
            com_y += np.random.normal(0, self.sensor_noise_std * 0.1)
            joint_pos += np.random.normal(0, self.sensor_noise_std * 0.1, size=joint_pos.shape)
            joint_vel += np.random.normal(0, self.sensor_noise_std * 0.2, size=joint_vel.shape)
            torso_linacc += np.random.normal(0, self.sensor_noise_std * 0.5, size=torso_linacc.shape)
        
        return {
            "roll": float(roll),
            "pitch": float(pitch),
            "roll_velocity": float(roll_velocity),
            "pitch_velocity": float(pitch_velocity),
            "com_x": float(com_x),
            "com_y": float(com_y),
            "torso_height": float(torso_height),
            "joint_positions": joint_pos,
            "joint_velocities": joint_vel,
            "torso_linear_acc": torso_linacc,
            "step": self.step_count,
            "time": self.step_count * self.dt,
        }
    
    def close(self):
        """Close environment and viewer."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

