"""Configuration constants and thresholds for evaluation."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""
    
    # Freeze detection
    FREEZE_TIME_THRESHOLD: float = 2.0  # seconds (lowered for more sensitive detection)
    MOVE_EPS: float = 0.02  # meters - minimum movement to not be considered frozen (increased)
    
    # Stability computation
    STABILITY_WINDOW_SIZE: int = 100  # steps for rolling stability window
    
    # Tilt zones for stability assessment
    # SAFE: Normal walking, small tilt
    # PREFALL: Significant tilt but recoverable (EDON should act aggressively here)
    # FAIL: Intervention threshold (same for baseline and EDON)
    SAFE_LIMIT: float = 0.15  # radians (~8.6 degrees) - safe zone limit
    PREFALL_LIMIT: float = 0.30  # radians (~17.2 degrees) - near-fall zone limit
    FAIL_LIMIT: float = 0.35  # radians (~20 degrees) - actual intervention threshold
    
    # Intervention detection (uses FAIL_LIMIT)
    # Note: These are set to FAIL_LIMIT to ensure consistency
    # Intervention is triggered when abs(roll) > FAIL_LIMIT OR abs(pitch) > FAIL_LIMIT
    FALL_THRESHOLD_ROLL: float = 0.35  # radians - same as FAIL_LIMIT
    FALL_THRESHOLD_PITCH: float = 0.35  # radians - same as FAIL_LIMIT
    ROLL_STD_THRESHOLD: float = 0.15  # radians - std threshold for instability
    SAFETY_TORQUE_LIMIT: float = 100.0  # Nm (adjust based on robot)
    SAFETY_JOINT_LIMIT: float = 3.14  # radians (adjust based on robot)
    
    # Episode limits
    MAX_EPISODE_STEPS: int = 10000
    MAX_EPISODE_TIME: float = 300.0  # seconds
    
    # Success criteria
    SUCCESS_DISTANCE_THRESHOLD: float = 0.1  # meters from goal
    SUCCESS_TIME_THRESHOLD: float = 60.0  # seconds to complete task
    
    # Environment randomization
    RANDOMIZE_FRICTION: bool = True
    FRICTION_MIN: float = 0.3
    FRICTION_MAX: float = 1.2
    
    RANDOMIZE_PUSHES: bool = True
    PUSH_FORCE_MIN: float = 10.0  # N (increased minimum)
    PUSH_FORCE_MAX: float = 150.0  # N (increased maximum for more challenge)
    PUSH_PROBABILITY: float = 0.15  # per step (increased frequency)
    
    RANDOMIZE_SENSOR_NOISE: bool = True
    SENSOR_NOISE_STD: float = 0.02  # Increased noise for more challenge
    
    # EDON integration
    EDON_BASE_URL: str = "http://127.0.0.1:8001"
    EDON_GRPC_PORT: int = 50052
    EDON_USE_GRPC: bool = False
    EDON_WINDOW_SIZE: int = 240  # 4 seconds @ 60Hz
    EDON_GAIN: float = 0.5  # How strongly EDON influences actions (0.0-1.0)
    
    # Logging
    LOG_CSV: bool = True
    LOG_JSON: bool = True
    LOG_PER_STEP: bool = False  # Set to True for detailed step-by-step logs
    
    # Stress profile (set via CLI, defaults to medium_stress if randomization enabled)
    STRESS_PROFILE: Optional[str] = None  # "light_stress", "medium_stress", "high_stress", "hell_stress"


# Global config instance
config = EvaluationConfig()

