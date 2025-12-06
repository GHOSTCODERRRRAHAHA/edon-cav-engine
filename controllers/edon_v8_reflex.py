"""
EDON v8 Reflex Layer Controller

Deterministic, fast micro-stabilization layer that applies smooth damping
and guardrails based on current state and fail-risk.
"""

from typing import Dict, Any, Optional
import numpy as np


class EdonReflexController:
    """
    Reflex layer for EDON v8.
    
    Takes baseline action and applies deterministic adjustments:
    - Adds damping when tilt/velocity is high
    - Clamps extreme commands
    - Increases damping when fail_risk is high
    """
    
    def __init__(
        self,
        max_damping_factor: float = 1.3,  # Reduced from 1.5 to allow more action
        fail_risk_damping_scale: float = 1.5,  # Reduced from 2.0
        tilt_damping_threshold: float = 0.15,
        vel_damping_threshold: float = 1.0
    ):
        """
        Initialize reflex controller.
        
        Args:
            max_damping_factor: Maximum damping multiplier (default: 1.5)
            fail_risk_damping_scale: How much fail_risk increases damping (default: 2.0)
            tilt_damping_threshold: Tilt magnitude threshold for damping (default: 0.15 rad)
            vel_damping_threshold: Velocity threshold for damping (default: 1.0 rad/s)
        """
        self.max_damping_factor = max_damping_factor
        self.fail_risk_damping_scale = fail_risk_damping_scale
        self.tilt_damping_threshold = tilt_damping_threshold
        self.vel_damping_threshold = vel_damping_threshold
    
    def compute_action(
        self,
        baseline_action: np.ndarray,
        obs: Dict[str, Any],
        fail_risk: float = 0.0,
        strategy_modulations: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Compute reflex-adjusted action.
        
        Args:
            baseline_action: Baseline control action (10-dim array)
            obs: Current observation dict
            fail_risk: Predicted failure risk ∈ [0, 1]
            strategy_modulations: Optional modulations from strategy layer:
                - gain_scale: Action scaling factor
                - lateral_compliance: Lateral compliance factor
                - step_height_bias: Step height adjustment
        
        Returns:
            Adjusted action array
        """
        action = baseline_action.copy()
        
        # Extract state
        roll = abs(obs.get("roll", 0.0))
        pitch = abs(obs.get("pitch", 0.0))
        roll_velocity = abs(obs.get("roll_velocity", 0.0))
        pitch_velocity = abs(obs.get("pitch_velocity", 0.0))
        
        tilt_mag = np.sqrt(roll**2 + pitch**2)
        vel_mag = np.sqrt(roll_velocity**2 + pitch_velocity**2)
        
        # Compute damping factor
        # Base damping from tilt
        tilt_damping = 1.0
        if tilt_mag > self.tilt_damping_threshold:
            # Increase damping as tilt increases
            excess_tilt = tilt_mag - self.tilt_damping_threshold
            tilt_damping = 1.0 + min(self.max_damping_factor - 1.0, excess_tilt * 2.0)
        
        # Additional damping from velocity
        vel_damping = 1.0
        if vel_mag > self.vel_damping_threshold:
            excess_vel = vel_mag - self.vel_damping_threshold
            vel_damping = 1.0 + min(self.max_damping_factor - 1.0, excess_vel * 0.5)
        
        # Fail-risk damping (strong effect when risk is high)
        fail_risk_damping = 1.0 + (fail_risk * self.fail_risk_damping_scale)
        fail_risk_damping = min(self.max_damping_factor, fail_risk_damping)
        
        # Combine damping factors (use max instead of multiply to be less aggressive)
        # This prevents compounding damping effects
        total_damping = min(self.max_damping_factor, max(tilt_damping, vel_damping, fail_risk_damping))
        
        # Apply damping (reduce action magnitude)
        # Damping > 1.0 means we reduce action, so we divide
        action = action / total_damping
        
        # Apply strategy modulations if provided
        if strategy_modulations:
            gain_scale = strategy_modulations.get("gain_scale", 1.0)
            action = action * gain_scale
            
            # Lateral compliance (affects lateral components, indices 0-3 typically)
            lateral_compliance = strategy_modulations.get("lateral_compliance", 1.0)
            if len(action) >= 4:
                action[:4] = action[:4] * lateral_compliance
            
            # Step height bias (affects vertical components, indices 4-7 typically)
            step_height_bias = strategy_modulations.get("step_height_bias", 0.0)
            if len(action) >= 8:
                action[4:8] = action[4:8] + step_height_bias * 0.1  # Small adjustment
        
        # Clamp to valid range
        action = np.clip(action, -1.0, 1.0)
        
        return action

