"""Humanoid simulation runner with metrics tracking."""

import time
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass

from evaluation.metrics import EpisodeMetrics, compute_stability_score
from evaluation.config import config
from evaluation.randomization import EnvironmentRandomizer


class HumanoidRunner:
    """
    Runner for humanoid simulation with metrics tracking.
    
    This class wraps a humanoid environment and tracks metrics during episodes.
    """
    
    def __init__(
        self,
        env: Any,
        controller: Callable,
        edon_client: Optional[Any] = None,
        randomizer: Optional[EnvironmentRandomizer] = None,
        use_edon: bool = False
    ):
        """
        Initialize runner.
        
        Args:
            env: Humanoid environment (gym-like interface)
            controller: Control policy function(obs, edon_state=None) -> action
            edon_client: EDON client (if use_edon=True)
            randomizer: Environment randomizer
            use_edon: Whether to use EDON in control loop
        """
        self.env = env
        self.controller = controller
        self.edon_client = edon_client
        self.randomizer = randomizer or EnvironmentRandomizer()
        self.use_edon = use_edon
        
        # Sensor window buffer for EDON
        self.sensor_window_buffer: List[Dict[str, Any]] = []
        self.sensor_window_size = config.EDON_WINDOW_SIZE
    
    def run_episode(
        self,
        episode_id: int,
        max_steps: Optional[int] = None,
        render: bool = False
    ) -> EpisodeMetrics:
        """
        Run a single episode and return metrics.
        
        Args:
            episode_id: Episode identifier
            max_steps: Maximum steps (None = use config default)
            render: Whether to render the episode
        
        Returns:
            EpisodeMetrics for this episode
        """
        max_steps = max_steps or config.MAX_EPISODE_STEPS
        
        # Initialize metrics
        metrics = EpisodeMetrics(episode_id=episode_id)
        metrics.metadata["max_steps"] = max_steps
        
        # Reset environment
        obs = self.env.reset()
        if self.randomizer:
            self.randomizer.apply_randomization(self.env)
        
        # Initialize tracking
        roll_history: List[float] = []
        pitch_history: List[float] = []
        com_history: List[float] = []
        tilt_zone_history: List[str] = []
        last_com = None
        freeze_start_time: Optional[float] = None
        episode_start_time = time.time()
        dt = 0.01  # Time step (assume 100Hz control loop)
        
        # Reset sensor buffer
        self.sensor_window_buffer = []
        
        step = 0
        done = False
        
        while not done and step < max_steps:
            step += 1
            current_time = time.time() - episode_start_time
            
            # Inject sensor noise if enabled
            if self.randomizer:
                obs = self.randomizer.inject_sensor_noise(obs)
                self.randomizer.apply_step_randomization(self.env, step)
            
            # Extract stability metrics from observation
            roll, pitch, com = self._extract_stability_metrics(obs)
            roll_history.append(roll)
            pitch_history.append(pitch)
            com_history.append(com)
            
            # Classify tilt zone and track metrics
            tilt_zone = self._classify_tilt_zone(roll, pitch)
            tilt_zone_history.append(tilt_zone)
            
            # Track time spent in each zone
            if tilt_zone == "safe":
                metrics.safe_time += dt
            elif tilt_zone == "prefall":
                metrics.prefall_events += 1
                metrics.prefall_time += dt
                metrics.prefall_times.append(current_time)
            elif tilt_zone == "fail":
                metrics.fail_events += 1
            
            # Detect freeze (enhanced definition)
            # Freeze = movement < MOVE_EPS for > FREEZE_TIME_THRESHOLD
            # OR speed almost zero while commanded to move (repeated minimal motion)
            if last_com is not None:
                movement = abs(com - last_com)
                
                # Check for minimal movement
                if movement < config.MOVE_EPS:
                    if freeze_start_time is None:
                        freeze_start_time = current_time
                    elif current_time - freeze_start_time > config.FREEZE_TIME_THRESHOLD:
                        # Freeze detected
                        metrics.freeze_events += 1
                        metrics.freeze_times.append(current_time)
                        freeze_start_time = None  # Reset after counting
                else:
                    freeze_start_time = None  # Reset if moving
                
                # Also check for repeated minimal motion (hesitation)
                # If movement is consistently very small over multiple steps
                if step > 5:  # Need some history
                    recent_movements = [abs(com_history[-i] - com_history[-i-1]) 
                                      for i in range(1, min(6, len(com_history)))]
                    if all(m < config.MOVE_EPS * 1.5 for m in recent_movements):
                        # Hesitation detected (not full freeze, but worth tracking)
                        if "hesitation_count" not in metrics.metadata:
                            metrics.metadata["hesitation_count"] = 0
                        metrics.metadata["hesitation_count"] += 1
            
            last_com = com
            
            # Detect interventions (fall, safety violations)
            # Note: Intervention is triggered at FAIL_LIMIT (0.35 rad)
            # This is the same threshold for baseline and EDON (fair comparison)
            intervention = self._detect_intervention(obs, roll, pitch)
            if intervention:
                metrics.interventions += 1
                metrics.intervention_times.append(current_time)
                # Note: fail_events is already incremented above if tilt_zone == "fail"
                # Optionally reset episode on intervention
                # For now, we continue but count it
            
            # Get EDON state if enabled
            edon_state = None
            if self.use_edon and self.edon_client:
                edon_state = self._get_edon_state(obs, step)
            
            # Get action from controller
            action = self.controller(obs, edon_state)
            
            # Step environment
            obs, reward, done, info = self.env.step(action)
            
            # Check for success (if defined in info)
            if info.get("success", False) or info.get("is_success", False):
                metrics.success = True
                metrics.metadata["success_time"] = current_time
            
            # Render if requested
            if render:
                self.env.render()
        
        # Compute final metrics
        metrics.episode_length = step
        metrics.episode_time = time.time() - episode_start_time
        metrics.roll_history = roll_history
        metrics.pitch_history = pitch_history
        metrics.com_history = com_history
        metrics.tilt_zone_history = tilt_zone_history
        metrics.stability_score = compute_stability_score(roll_history, pitch_history)
        
        # Compute enhanced stability metrics
        if len(roll_history) > 0:
            roll_arr = np.array(roll_history)
            metrics.roll_rms = float(np.sqrt(np.mean(roll_arr**2)))
            metrics.roll_max = float(np.max(np.abs(roll_arr)))
            metrics.roll_std = float(np.std(roll_arr))
        
        if len(pitch_history) > 0:
            pitch_arr = np.array(pitch_history)
            metrics.pitch_rms = float(np.sqrt(np.mean(pitch_arr**2)))
            metrics.pitch_max = float(np.max(np.abs(pitch_arr)))
            metrics.pitch_std = float(np.std(pitch_arr))
        
        if len(com_history) > 0:
            com_arr = np.array(com_history)
            metrics.com_deviation = float(np.sqrt(np.mean(com_arr**2)))
        
        # Store additional metadata
        metrics.metadata["done"] = done
        metrics.metadata["final_reward"] = info.get("reward", 0.0) if 'info' in locals() else 0.0
        
        return metrics
    
    def _extract_stability_metrics(self, obs: Dict[str, Any]) -> tuple:
        """
        Extract roll, pitch, and center of mass from observation.
        
        Adapt this to your environment's observation space.
        Common field names:
        - roll, pitch, yaw
        - torso_roll, torso_pitch
        - com_x, com_y, com_z
        - center_of_mass
        """
        # Try common field names
        roll = obs.get("roll", obs.get("torso_roll", obs.get("orientation_roll", 0.0)))
        pitch = obs.get("pitch", obs.get("torso_pitch", obs.get("orientation_pitch", 0.0)))
        
        # Center of mass - try different representations
        if "com_x" in obs and "com_y" in obs:
            com = np.sqrt(obs["com_x"]**2 + obs["com_y"]**2)
        elif "center_of_mass" in obs:
            com_arr = obs["center_of_mass"]
            if isinstance(com_arr, (list, np.ndarray)) and len(com_arr) >= 2:
                com = np.sqrt(com_arr[0]**2 + com_arr[1]**2)
            else:
                com = 0.0
        elif "com" in obs:
            com = obs["com"]
        else:
            # Fallback: use position if available
            if "x" in obs and "y" in obs:
                com = np.sqrt(obs["x"]**2 + obs["y"]**2)
            else:
                com = 0.0
        
        return float(roll), float(pitch), float(com)
    
    def _detect_intervention(self, obs: Dict[str, Any], roll: float, pitch: float) -> bool:
        """
        Detect if intervention is needed (fall, safety violation).
        
        Tilt Computation:
        - Roll and pitch are extracted from observation (torso orientation angles in radians)
        - Tilt magnitude = sqrt(roll^2 + pitch^2)
        - Intervention is triggered when tilt exceeds FAIL_LIMIT threshold
        
        Intervention Triggers:
        1. Fall: abs(roll) > FAIL_LIMIT OR abs(pitch) > FAIL_LIMIT
           - FAIL_LIMIT = 0.35 rad (~20 degrees)
           - This is the same threshold for both baseline and EDON (fair comparison)
        2. Torque violation: max_torque > SAFETY_TORQUE_LIMIT
        3. Joint limit violation: max_joint_angle > SAFETY_JOINT_LIMIT
        
        Returns:
            True if intervention detected
        """
        # Check for fall (excessive roll/pitch) - uses FAIL_LIMIT
        if abs(roll) > config.FALL_THRESHOLD_ROLL or abs(pitch) > config.FALL_THRESHOLD_PITCH:
            return True
        
        # Check for torque violations (if available in obs)
        if "torque" in obs:
            torque = obs["torque"]
            if isinstance(torque, (list, np.ndarray)):
                max_torque = np.max(np.abs(torque))
                if max_torque > config.SAFETY_TORQUE_LIMIT:
                    return True
            elif abs(torque) > config.SAFETY_TORQUE_LIMIT:
                return True
        
        # Check for joint limit violations (if available)
        if "joint_angles" in obs:
            joint_angles = obs["joint_angles"]
            if isinstance(joint_angles, (list, np.ndarray)):
                max_angle = np.max(np.abs(joint_angles))
                if max_angle > config.SAFETY_JOINT_LIMIT:
                    return True
        
        return False
    
    def _classify_tilt_zone(self, roll: float, pitch: float) -> str:
        """
        Classify current tilt into safe/prefall/fail zones.
        
        Zones:
        - SAFE: tilt_magnitude <= SAFE_LIMIT (0.15 rad) - normal walking
        - PREFALL: SAFE_LIMIT < tilt_magnitude <= PREFALL_LIMIT (0.30 rad) - near-fall, recoverable
        - FAIL: tilt_magnitude > FAIL_LIMIT (0.35 rad) - intervention triggered
        
        Args:
            roll: Roll angle (radians)
            pitch: Pitch angle (radians)
        
        Returns:
            "safe", "prefall", or "fail"
        """
        tilt_magnitude = np.sqrt(roll**2 + pitch**2)
        max_tilt = max(abs(roll), abs(pitch))
        
        # Use max of magnitude or individual axis for classification
        # (intervention uses individual axis, so we should too)
        if max_tilt > config.FAIL_LIMIT:
            return "fail"
        elif max_tilt > config.PREFALL_LIMIT or tilt_magnitude > config.PREFALL_LIMIT:
            return "prefall"
        elif max_tilt > config.SAFE_LIMIT or tilt_magnitude > config.SAFE_LIMIT:
            return "prefall"  # Between safe and prefall limit
        else:
            return "safe"
    
    def _get_edon_state(self, obs: Dict[str, Any], step: int) -> Optional[Dict[str, Any]]:
        """
        Get EDON state from current observation.
        
        Builds sensor window and calls EDON client.
        """
        if not self.edon_client:
            return None
        
        # Build sensor window from observation
        # This is a simplified version - adapt to your observation space
        window = self._build_sensor_window(obs)
        
        # Add to buffer
        self.sensor_window_buffer.append(window)
        
        # Keep only last N windows
        if len(self.sensor_window_buffer) > self.sensor_window_size:
            self.sensor_window_buffer.pop(0)
        
        # Only call EDON when we have enough data
        if len(self.sensor_window_buffer) < self.sensor_window_size:
            return None
        
        try:
            # Use v2 API if available
            if hasattr(self.edon_client, 'cav_batch_v2'):
                response = self.edon_client.cav_batch_v2(windows=[window])
                if isinstance(response, dict) and "results" in response:
                    result = response["results"][0]
                else:
                    result = response[0] if isinstance(response, list) else response
            else:
                # Fallback to v1 API
                response = self.edon_client.cav(window)
                result = response
            
            # Extract state information
            return {
                "state_class": result.get("state_class", result.get("state", "unknown")),
                "p_stress": result.get("p_stress", 0.0),
                "p_chaos": result.get("p_chaos", 0.0),
                "influences": result.get("influences", {}),
                "confidence": result.get("confidence", 1.0),
                "cav_vector": result.get("cav_vector")  # Include CAV vector for controller
            }
        except Exception as e:
            # Log error but don't crash
            print(f"Warning: EDON call failed: {e}")
            return None
    
    def _build_sensor_window(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build EDON sensor window from observation.
        
        This is a stub - adapt to your observation space.
        EDON expects 240 samples per signal (4 seconds @ 60Hz).
        For real-time, you'd buffer samples. For simulation, we can use
        current values repeated or interpolate.
        """
        # Extract sensor values from observation
        # This is simplified - in reality you'd buffer actual sensor readings
        
        # Get current values
        acc_x = obs.get("acc_x", obs.get("acceleration_x", 0.0))
        acc_y = obs.get("acc_y", obs.get("acceleration_y", 0.0))
        acc_z = obs.get("acc_z", obs.get("acceleration_z", 1.0))
        
        # For simulation, we'll use current values with some variation
        # In real robot, you'd have actual sensor buffers
        samples = self.sensor_window_size
        
        # Create v2 window format
        window = {
            "physio": {
                "EDA": [0.1] * samples,  # Stub - would come from actual sensors
                "BVP": [0.5] * samples   # Stub
            },
            "motion": {
                "ACC_x": [acc_x] * samples,
                "ACC_y": [acc_y] * samples,
                "ACC_z": [acc_z] * samples
            },
            "env": {
                "temp_c": obs.get("temp_c", 22.0),
                "humidity": obs.get("humidity", 45.0),
                "aqi": obs.get("aqi", 20)
            },
            "task": {
                "id": obs.get("task_id", "simulation"),
                "complexity": obs.get("task_complexity", 0.5)
            }
        }
        
        return window

