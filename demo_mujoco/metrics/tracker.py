"""Track and compute stability metrics."""

import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque


class MetricsTracker:
    """
    Track stability metrics during episode execution.
    
    Metrics:
    - Falls per episode
    - Freezes per episode
    - Interventions per episode
    - Recovery time after disturbances
    - Stability score
    """
    
    def __init__(
        self,
        fall_height_threshold: float = 0.3,
        fall_tilt_threshold: float = 0.5,  # radians (~30 degrees)
        freeze_velocity_threshold: float = 0.01,
        freeze_duration_threshold: float = 2.0,
        stability_window_size: int = 100
    ):
        """
        Initialize metrics tracker.
        
        Args:
            fall_height_threshold: Torso height below which is considered a fall
            fall_tilt_threshold: Tilt angle (radians) above which is considered a fall
            freeze_velocity_threshold: Velocity magnitude below which is considered frozen
            freeze_duration_threshold: Duration (seconds) of low velocity to count as freeze
            stability_window_size: Window size for computing stability score
        """
        self.fall_height_threshold = fall_height_threshold
        self.fall_tilt_threshold = fall_tilt_threshold
        self.freeze_velocity_threshold = freeze_velocity_threshold
        self.freeze_duration_threshold = freeze_duration_threshold
        self.stability_window_size = stability_window_size
        
        # Episode state
        self.reset()
    
    def reset(self):
        """Reset metrics for new episode."""
        self.falls = 0
        self.freezes = 0
        self.interventions = 0
        self.recovery_times: List[float] = []
        self.stability_scores: deque = deque(maxlen=self.stability_window_size)
        
        # State tracking
        self.last_disturbance_time: Optional[float] = None
        self.freeze_start_time: Optional[float] = None
        self.fallen = False
        self.frozen = False
        self._last_intervention_step: Optional[int] = None
        self._in_intervention_state = False  # Track if we're currently in intervention state
        
        # History for stability computation
        self.roll_history: deque = deque(maxlen=self.stability_window_size)
        self.pitch_history: deque = deque(maxlen=self.stability_window_size)
        self.com_velocity_history: deque = deque(maxlen=self.stability_window_size)
    
    def update(
        self,
        state: Dict[str, Any],
        info: Dict[str, Any],
        dt: float = 0.01
    ):
        """
        Update metrics with new state.
        
        Args:
            state: Current robot state observation
            info: Info dict from environment step
            dt: Timestep
        """
        current_time = state.get("time", 0.0)
        step_count = info.get("step", 0)  # Get step count from info
        torso_height = state.get("torso_height", 1.0)
        roll = abs(state.get("roll", 0.0))
        pitch = abs(state.get("pitch", 0.0))
        roll_vel = state.get("roll_velocity", 0.0)
        pitch_vel = state.get("pitch_velocity", 0.0)
        com_x = state.get("com_x", 0.0)
        com_y = state.get("com_y", 0.0)
        
        # Check for fall (more lenient - only count as fall if really unstable)
        tilt = max(roll, pitch)
        # Only count as fall if both height is low AND tilt is high (more conservative)
        if not self.fallen:
            if torso_height < self.fall_height_threshold and tilt > self.fall_tilt_threshold * 0.7:  # 70% of threshold
                self.falls += 1
                self.fallen = True
            elif torso_height < self.fall_height_threshold * 0.5:  # Very low height
                self.falls += 1
                self.fallen = True
            elif tilt > self.fall_tilt_threshold * 1.5:  # Extreme tilt
                self.falls += 1
                self.fallen = True
        
        # Check for freeze
        velocity_magnitude = np.sqrt(roll_vel**2 + pitch_vel**2)
        if velocity_magnitude < self.freeze_velocity_threshold:
            if self.freeze_start_time is None:
                self.freeze_start_time = current_time
            elif current_time - self.freeze_start_time >= self.freeze_duration_threshold:
                if not self.frozen:
                    self.freezes += 1
                    self.frozen = True
        else:
            self.freeze_start_time = None
            self.frozen = False
        
        # Check for intervention (matches original environment)
        # Intervention is detected when tilt exceeds 0.35 rad (~20 degrees)
        # This matches the original FAIL_LIMIT threshold
        # Count intervention as an EVENT (entering intervention state), not every step while in it
        intervention_detected = info.get("intervention_detected", False) or info.get("intervention", False)
        
        if intervention_detected:
            # Only count when ENTERING intervention state (not every step while in it)
            if not self._in_intervention_state:
                self.interventions += 1
                self._in_intervention_state = True
                self._last_intervention_step = step_count
                # Diagnostic: Log intervention details (first few only to avoid spam)
                if self.interventions <= 3:
                    print(f"  [Metrics] Intervention #{self.interventions} detected at step {step_count}: "
                          f"roll={roll:.3f} rad, pitch={pitch:.3f} rad (threshold=0.35 rad)")
        else:
            # Reset intervention state when we exit
            self._in_intervention_state = False
        
        # Track disturbance recovery
        if info.get("disturbance_active", False):
            self.last_disturbance_time = current_time
        
        # Check recovery after disturbance
        if (self.last_disturbance_time is not None and
            current_time > self.last_disturbance_time and
            tilt < 0.1 and velocity_magnitude < 0.1):
            # Recovered
            recovery_time = current_time - self.last_disturbance_time
            if recovery_time > 0 and recovery_time < 10.0:  # Reasonable recovery time
                self.recovery_times.append(recovery_time)
            self.last_disturbance_time = None
        
        # Update stability score history
        self.roll_history.append(roll)
        self.pitch_history.append(pitch)
        com_velocity = np.sqrt(com_x**2 + com_y**2)
        self.com_velocity_history.append(com_velocity)
        
        # Compute current stability score
        if len(self.roll_history) > 10:
            roll_variance = np.var(list(self.roll_history))
            pitch_variance = np.var(list(self.pitch_history))
            com_velocity_mean = np.mean(list(self.com_velocity_history))
            
            # Stability score: negative of variance (lower variance = higher stability)
            stability = -(roll_variance + pitch_variance) - 0.1 * com_velocity_mean
            self.stability_scores.append(stability)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics summary.
        
        Returns:
            Dict with:
            - falls: int
            - freezes: int
            - interventions: int
            - avg_recovery_time: float
            - stability_score: float
        """
        avg_recovery = np.mean(self.recovery_times) if self.recovery_times else 0.0
        stability_score = np.mean(list(self.stability_scores)) if self.stability_scores else 0.0
        
        return {
            "falls": self.falls,
            "freezes": self.freezes,
            "interventions": self.interventions,
            "avg_recovery_time": float(avg_recovery),
            "stability_score": float(stability_score),
        }
    
    def get_live_metrics(self) -> Dict[str, Any]:
        """
        Get live metrics for display (updated every step).
        
        Returns:
            Dict with current state metrics
        """
        current_roll = list(self.roll_history)[-1] if self.roll_history else 0.0
        current_pitch = list(self.pitch_history)[-1] if self.pitch_history else 0.0
        current_stability = list(self.stability_scores)[-1] if self.stability_scores else 0.0
        
        return {
            "current_roll": float(current_roll),
            "current_pitch": float(current_pitch),
            "current_stability": float(current_stability),
            "fallen": self.fallen,
            "frozen": self.frozen,
            **self.get_metrics()
        }

