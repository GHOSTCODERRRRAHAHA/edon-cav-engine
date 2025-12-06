"""Baseline controller with balance control layer for better EDON effectiveness."""

import numpy as np
from typing import Dict, Any


class BaselineController:
    """
    Baseline controller with balance control layer.
    
    Maps balance control (roll/pitch/COM) to joint torques for better EDON effectiveness.
    This creates a more direct control abstraction similar to the original environment.
    """
    
    def __init__(self, action_size: int = 12):
        """
        Initialize baseline controller.
        
        Args:
            action_size: Size of action vector (default: 12 for MuJoCo)
        """
        self.action_size = action_size
        
        # Balance control gains (same as original)
        self.kp_roll = 0.5
        self.kp_pitch = 0.5
        self.kp_com_x = 0.3
        self.kp_com_y = 0.3
        
        # Mapping from balance control to joint torques
        # This creates a more direct abstraction
        self._init_balance_to_joint_mapping()
    
    def _init_balance_to_joint_mapping(self):
        """Initialize mapping from balance control to joint torques."""
        # Create a simple mapping: balance corrections → joint torques
        # Root rotation joints (indices 3-5) for roll/pitch control
        # Root position joints (indices 0-2) for COM control
        # This makes EDON's modulations more effective
        pass
    
    def step(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Compute control action from state with balance control layer.
        
        This creates a balance control abstraction (like original) that maps to joint torques:
        1. Compute balance corrections (roll/pitch/COM) - same as original
        2. Map balance corrections to joint torques - new layer for MuJoCo
        3. Add exploration noise
        4. Scale to MuJoCo range
        
        Args:
            state: Observation dict with roll, pitch, COM, etc.
        
        Returns:
            Control action (torques for actuators in [-20, 20] range)
        """
        # Extract state (same as original)
        roll = state.get("roll", 0.0)
        pitch = state.get("pitch", 0.0)
        com_x = state.get("com_x", 0.0)
        com_y = state.get("com_y", 0.0)
        
        # STEP 1: Compute balance control corrections (same as original)
        # This is the "balance control layer" - direct roll/pitch/COM control
        balance_roll = -roll * self.kp_roll  # Correct roll
        balance_pitch = -pitch * self.kp_pitch  # Correct pitch
        balance_com_x = -com_x * self.kp_com_x  # Correct COM x
        balance_com_y = -com_y * self.kp_com_y  # Correct COM y
        
        # STEP 2: Map balance control to joint torques (balance control layer)
        # This creates a more direct abstraction for EDON
        action = np.zeros(self.action_size)
        
        # Map balance corrections to root joints (direct control)
        # Root position (indices 0-2) for COM control
        action[0] = balance_com_x  # root_x
        action[1] = balance_com_y  # root_y
        action[2] = 0.0  # root_z (height)
        
        # Root rotation (indices 3-5) for roll/pitch control
        action[3] = balance_roll  # root_rot_x
        action[4] = balance_pitch  # root_rot_y
        action[5] = 0.0  # root_rot_z
        
        # Joint control (indices 6-11) - maintain nominal pose
        # Use small corrections to maintain standing
        joint_positions = state.get("joint_positions", np.zeros(6))
        target_joint_pos = np.zeros(6)  # Nominal standing pose
        
        for i in range(6):
            joint_error = target_joint_pos[i] - joint_positions[i]
            # Small joint corrections (much smaller than balance control)
            action[6 + i] = joint_error * 0.1
        
        # Add some exploration noise (makes baseline less stable - original behavior)
        action += np.random.normal(0, 0.1, size=self.action_size)
        
        # Clip to original range [-1.0, 1.0] (as in original)
        action = np.clip(action, -1.0, 1.0)
        
        # Scale to MuJoCo action range [-20, 20]
        # Original uses [-1, 1], MuJoCo needs [-20, 20]
        action = action * 20.0
        
        return action

