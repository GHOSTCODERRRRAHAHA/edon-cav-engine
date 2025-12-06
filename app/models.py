"""Pydantic models for EDON CAV Engine API."""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional


WINDOW_LEN = 240  # 60 seconds at 4 Hz


class CAVRequest(BaseModel):
    """Request model for CAV computation."""
    
    EDA: List[float] = Field(..., description="EDA signal (240 samples)")
    TEMP: List[float] = Field(..., description="Temperature signal (240 samples)")
    BVP: List[float] = Field(..., description="Blood volume pulse signal (240 samples)")
    ACC_x: List[float] = Field(..., description="Accelerometer X-axis (240 samples)")
    ACC_y: List[float] = Field(..., description="Accelerometer Y-axis (240 samples)")
    ACC_z: List[float] = Field(..., description="Accelerometer Z-axis (240 samples)")
    temp_c: float = Field(..., description="Environmental temperature in Celsius")
    humidity: float = Field(..., description="Humidity percentage")
    aqi: int = Field(..., description="Air Quality Index")
    local_hour: int = Field(12, ge=0, le=23, description="Local hour [0-23]")
    
    @validator('EDA', 'TEMP', 'BVP', 'ACC_x', 'ACC_y', 'ACC_z')
    def validate_array_length(cls, v):
        """Validate that each signal array has exactly 240 samples."""
        if len(v) != WINDOW_LEN:
            raise ValueError(f"Array must have exactly {WINDOW_LEN} elements, got {len(v)}")
        return v


class AdaptiveInfo(BaseModel):
    """Adaptive adjustment information."""
    
    z_cav: float = Field(..., description="Z-score of current CAV relative to baseline")
    sensitivity: float = Field(..., description="Sensitivity multiplier (1.0 = normal, >1.0 = increased)")
    env_weight_adj: float = Field(..., description="Environment weight adjustment (1.0 = normal, <1.0 = reduced)")


class CAVResponse(BaseModel):
    """Response model for CAV computation."""
    
    cav_raw: int = Field(..., description="Raw CAV score [0-10000]")
    cav_smooth: int = Field(..., description="EMA-smoothed CAV score [0-10000]")
    state: str = Field(..., description="State: overload, balanced, focus, or restorative")
    parts: Dict[str, float] = Field(..., description="Component scores: bio, env, circadian, p_stress")
    adaptive: Optional[AdaptiveInfo] = Field(None, description="Adaptive adjustments (if memory engine enabled)")


class BatchRequest(BaseModel):
    """Request model for batch CAV computation."""
    
    windows: List[CAVRequest] = Field(..., description="List of sensor windows to process")


class BatchResponseItem(BaseModel):
    """Single item in batch response."""
    
    ok: bool = True
    error: Optional[str] = None
    
    cav_raw: Optional[int] = None
    cav_smooth: Optional[int] = None
    state: Optional[str] = None
    parts: Optional[Dict[str, float]] = None


class BatchResponse(BaseModel):
    """Response model for batch CAV computation."""
    
    results: List[BatchResponseItem] = Field(..., description="CAV results for each window")
    latency_ms: float = Field(..., description="Total processing latency in milliseconds")
    server_version: str = Field(..., description="Server version string")


class HealthResponse(BaseModel):
    """Health check response."""
    
    ok: bool = Field(..., description="Service health status")
    model: str = Field(..., description="Model identifier")
    uptime_s: Optional[float] = Field(None, description="Server uptime in seconds")


class TelemetryResponse(BaseModel):
    """Telemetry response."""
    
    request_count: int = Field(..., description="Total number of requests processed")
    avg_latency_ms: float = Field(..., description="Average latency in milliseconds")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")


# ============================================================================
# Robot Stability Models (v8 Integration)
# ============================================================================

class RobotState(BaseModel):
    """Robot state input for stability control."""
    
    roll: float = Field(..., description="Roll angle (radians)")
    pitch: float = Field(..., description="Pitch angle (radians)")
    roll_velocity: float = Field(..., description="Roll angular velocity (rad/s)")
    pitch_velocity: float = Field(..., description="Pitch angular velocity (rad/s)")
    com_x: float = Field(0.0, description="Center of mass X position")
    com_y: float = Field(0.0, description="Center of mass Y position")
    # Optional additional state
    com_z: Optional[float] = Field(None, description="Center of mass Z position")
    yaw: Optional[float] = Field(None, description="Yaw angle (radians)")
    yaw_velocity: Optional[float] = Field(None, description="Yaw angular velocity (rad/s)")


class Modulations(BaseModel):
    """Control modulations output."""
    
    gain_scale: float = Field(..., description="Gain scale multiplier [0.5-2.0]")
    compliance: float = Field(..., description="Lateral compliance [0.0-1.0]")
    bias: List[float] = Field(..., description="Step height bias vector (action space size)")


class RobotStabilityRequest(BaseModel):
    """Request model for robot stability control."""
    
    robot_state: RobotState = Field(..., description="Current robot state")
    history: Optional[List[RobotState]] = Field(None, description="Previous robot states (for temporal memory, max 8)")
    fail_risk: Optional[float] = Field(None, description="Pre-computed fail risk [0.0-1.0] (optional)")
    baseline_action: Optional[List[float]] = Field(None, description="Baseline controller action (optional, will compute if not provided)")


class RobotStabilityResponse(BaseModel):
    """Response model for robot stability control."""
    
    strategy_id: int = Field(..., description="Selected strategy ID [0-3]: 0=NORMAL, 1=HIGH_DAMPING, 2=RECOVERY_BALANCE, 3=COMPLIANT_TERRAIN")
    strategy_name: str = Field(..., description="Strategy name")
    modulations: Modulations = Field(..., description="Control modulations")
    intervention_risk: float = Field(..., description="Predicted intervention risk [0.0-1.0]")
    latency_ms: Optional[float] = Field(None, description="Processing latency in milliseconds")


class RecordOutcomeRequest(BaseModel):
    """Request model for recording intervention outcome."""
    
    strategy_id: int = Field(..., description="Strategy ID used")
    gain_scale: float = Field(..., description="Gain scale modulation used")
    lateral_compliance: float = Field(..., description="Lateral compliance modulation used")
    step_height_bias: float = Field(..., description="Step height bias modulation used")
    intervention_occurred: bool = Field(..., description="Whether intervention occurred")
    fail_risk: float = Field(..., description="Intervention risk at time of action")
    robot_state: Optional[RobotState] = Field(None, description="Robot state at time of action")