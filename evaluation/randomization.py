"""Environment randomization for uncontrolled environment simulation."""

import random
import numpy as np
from typing import Dict, Any, Optional, Callable
from evaluation.config import config


class EnvironmentRandomizer:
    """Handles environment randomization for robustness testing."""
    
    def __init__(self, seed: Optional[int] = None, stress_profile: Optional[Any] = None):
        """
        Initialize randomizer with optional seed and stress profile.
        
        Args:
            seed: Random seed for reproducibility
            stress_profile: StressProfile object (from evaluation.stress_profiles)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.seed = seed
        self.stress_profile = stress_profile
        self.friction_value: Optional[float] = None
        
        # Use stress profile settings if provided, otherwise use config defaults
        if stress_profile:
            self.sensor_noise_std = stress_profile.sensor_noise_std
            self.push_force_min = stress_profile.push_force_min
            self.push_force_max = stress_profile.push_force_max
            self.push_probability = stress_profile.push_probability
            self.friction_min = stress_profile.friction_min
            self.friction_max = stress_profile.friction_max
            self.actuator_delay_steps = stress_profile.actuator_delay_steps
            self.fatigue_enabled = stress_profile.fatigue_enabled
            self.fatigue_degradation = stress_profile.fatigue_degradation
        else:
            self.sensor_noise_std = config.SENSOR_NOISE_STD
            self.push_force_min = config.PUSH_FORCE_MIN
            self.push_force_max = config.PUSH_FORCE_MAX
            self.push_probability = config.PUSH_PROBABILITY
            self.friction_min = config.FRICTION_MIN
            self.friction_max = config.FRICTION_MAX
            self.actuator_delay_steps = (0, 0)
            self.fatigue_enabled = False
            self.fatigue_degradation = 0.0
        
        # Fatigue tracking
        self.episode_start_step = 0
        self.fatigue_factor = 1.0  # Starts at 1.0, decreases over episode
    
    def apply_randomization(self, env: Any) -> Dict[str, Any]:
        """
        Apply randomization to environment at episode reset.
        
        Args:
            env: Environment object (generic, adapt to your sim)
        
        Returns:
            Dictionary of applied randomizations
        """
        applied = {}
        
        # Reset fatigue
        self.episode_start_step = 0
        self.fatigue_factor = 1.0
        
        # Randomize friction (more extreme values)
        if config.RANDOMIZE_FRICTION:
            # Use beta distribution to favor extremes (low or high friction)
            if random.random() < 0.5:
                # Low friction (slippery)
                friction = random.uniform(self.friction_min, self.friction_min + 0.2)
            else:
                # High friction (sticky)
                friction = random.uniform(self.friction_max - 0.2, self.friction_max)
            applied["friction"] = friction
            self._set_friction(env, friction)
        
        # Add floor incline if supported (from stress profile)
        if self.stress_profile and hasattr(env, 'set_floor_incline'):
            incline_min, incline_max = self.stress_profile.floor_incline_range
            if incline_max > 0:
                incline = random.uniform(incline_min, incline_max)
                env.set_floor_incline(incline)
                applied["floor_incline"] = incline
        
        # Add height variation if supported
        if self.stress_profile and hasattr(env, 'set_height_variation'):
            height_min, height_max = self.stress_profile.height_variation_range
            if height_max > 0:
                height_var = random.uniform(height_min, height_max)
                env.set_height_variation(height_var)
                applied["height_variation"] = height_var
        
        # Reset push state
        applied["push_applied"] = False
        
        return applied
    
    def apply_step_randomization(self, env: Any, step: int) -> Dict[str, Any]:
        """
        Apply per-step randomization (pushes, noise, delays, fatigue).
        
        Args:
            env: Environment object
            step: Current step number
        
        Returns:
            Dictionary of applied randomizations
        """
        applied = {}
        
        # Update fatigue factor (if enabled)
        if self.fatigue_enabled and self.stress_profile:
            # Fatigue increases over episode (degradation factor decreases)
            # Assume episode length ~1000 steps, apply linear degradation
            episode_progress = step / 1000.0  # Normalize to [0, 1]
            self.fatigue_factor = 1.0 - (self.fatigue_degradation * episode_progress)
            applied["fatigue_factor"] = self.fatigue_factor
        
        # Random pushes (using stress profile settings)
        if config.RANDOMIZE_PUSHES:
            if random.random() < self.push_probability:
                # Use exponential distribution for occasional strong pushes
                if random.random() < 0.2:  # 20% chance of strong push
                    force = random.uniform(self.push_force_max * 0.7, self.push_force_max)
                else:
                    force = random.uniform(self.push_force_min, self.push_force_max * 0.5)
                direction = random.uniform(0, 2 * np.pi)  # Random direction
                applied["push"] = {"force": force, "direction": direction}
                self._apply_push(env, force, direction)
        
        # Add occasional sensor jitter (random delays in sensor readings)
        if random.random() < 0.05:  # 5% chance per step
            applied["sensor_jitter"] = True
        
        # Apply actuator delay (if stress profile has it)
        if self.stress_profile and self.actuator_delay_steps[1] > 0:
            # Delay is applied in the environment step, not here
            # This is just for tracking
            applied["actuator_delay"] = random.randint(
                self.actuator_delay_steps[0],
                self.actuator_delay_steps[1]
            )
        
        return applied
    
    def inject_sensor_noise(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject noise into sensor readings.
        
        Args:
            observation: Observation dictionary
        
        Returns:
            Observation with noise injected
        """
        if not config.RANDOMIZE_SENSOR_NOISE:
            return observation
        
        noisy_obs = observation.copy()
        
        # Add noise to common sensor fields (using stress profile noise level)
        noise_fields = ["roll", "pitch", "yaw", "acc_x", "acc_y", "acc_z",
                       "gyro_x", "gyro_y", "gyro_z", "com_x", "com_y", "com_z"]
        
        for field in noise_fields:
            if field in noisy_obs:
                if isinstance(noisy_obs[field], (list, np.ndarray)):
                    noise = np.random.normal(0, self.sensor_noise_std, len(noisy_obs[field]))
                    noisy_obs[field] = np.array(noisy_obs[field]) + noise
                else:
                    noise = np.random.normal(0, self.sensor_noise_std)
                    noisy_obs[field] = noisy_obs[field] + noise
        
        # Apply fatigue to sensor readings (if enabled)
        if self.fatigue_enabled and self.fatigue_factor < 1.0:
            # Fatigue makes sensors noisier
            fatigue_noise_boost = (1.0 - self.fatigue_factor) * 0.5  # Up to 50% more noise
            for field in noise_fields:
                if field in noisy_obs:
                    if isinstance(noisy_obs[field], (list, np.ndarray)):
                        fatigue_noise = np.random.normal(0, self.sensor_noise_std * fatigue_noise_boost, len(noisy_obs[field]))
                        noisy_obs[field] = np.array(noisy_obs[field]) + fatigue_noise
                    else:
                        fatigue_noise = np.random.normal(0, self.sensor_noise_std * fatigue_noise_boost)
                        noisy_obs[field] = noisy_obs[field] + fatigue_noise
        
        return noisy_obs
    
    def _set_friction(self, env: Any, friction: float) -> None:
        """
        Set friction in environment.
        
        This is a stub - adapt to your environment's API.
        Common patterns:
        - env.set_friction(friction)
        - env.model.geom_friction = friction
        - env.physics.model.geom_friction = friction
        """
        if hasattr(env, 'set_friction'):
            env.set_friction(friction)
        elif hasattr(env, 'model') and hasattr(env.model, 'geom_friction'):
            env.model.geom_friction[:] = friction
        elif hasattr(env, 'physics') and hasattr(env.physics.model, 'geom_friction'):
            env.physics.model.geom_friction[:] = friction
        # If none of these work, the environment doesn't support friction changes
        # This is fine - the randomization will be skipped
    
    def _apply_push(self, env: Any, force: float, direction: float) -> None:
        """
        Apply external push to robot.
        
        This is a stub - adapt to your environment's API.
        Common patterns:
        - env.apply_force(force, direction)
        - env.model.body_force = force_vector
        """
        if hasattr(env, 'apply_force'):
            env.apply_force(force, direction)
        elif hasattr(env, 'apply_external_force'):
            # Convert direction to force vector
            force_x = force * np.cos(direction)
            force_y = force * np.sin(direction)
            env.apply_external_force(force_x, force_y, 0.0)
        # If none of these work, the environment doesn't support external forces
        # This is fine - the randomization will be skipped

