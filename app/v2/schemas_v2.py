"""Pydantic schemas for EDON v2 multimodal API."""

from pydantic import BaseModel, Field, validator, model_validator
from typing import List, Dict, Optional, Any
import numpy as np


# Window length for physiological signals (240 samples = 4 seconds @ 60Hz)
WINDOW_LEN = 240


class PhysioInput(BaseModel):
    """Physiological signal inputs."""
    
    EDA: Optional[List[float]] = Field(None, description="Electrodermal activity (240 samples)")
    TEMP: Optional[List[float]] = Field(None, description="Temperature signal (240 samples)")
    BVP: Optional[List[float]] = Field(None, description="Blood volume pulse (240 samples)")
    
    @validator('EDA', 'TEMP', 'BVP')
    def validate_array_length(cls, v):
        """Validate that each signal array has exactly 240 samples if provided."""
        if v is not None and len(v) != WINDOW_LEN:
            raise ValueError(f"Array must have exactly {WINDOW_LEN} elements, got {len(v)}")
        return v


class MotionInput(BaseModel):
    """Motion and torque inputs."""
    
    # Accelerometer data (primary motion signal)
    ACC_x: Optional[List[float]] = Field(None, description="Accelerometer X-axis (240 samples)")
    ACC_y: Optional[List[float]] = Field(None, description="Accelerometer Y-axis (240 samples)")
    ACC_z: Optional[List[float]] = Field(None, description="Accelerometer Z-axis (240 samples)")
    
    # Additional motion signals (optional)
    velocity: Optional[List[float]] = Field(None, description="Velocity vector [vx, vy, vz] or time series")
    acceleration: Optional[List[float]] = Field(None, description="Acceleration vector [ax, ay, az] or time series")
    torque: Optional[List[float]] = Field(None, description="Torque measurements")
    joint_angles: Optional[List[float]] = Field(None, description="Joint angle measurements")
    force: Optional[List[float]] = Field(None, description="Force measurements")
    
    # Aggregated features (optional, if pre-computed)
    velocity_magnitude: Optional[float] = Field(None, description="Velocity magnitude")
    torque_mean: Optional[float] = Field(None, description="Mean torque")
    force_mean: Optional[float] = Field(None, description="Mean force")
    
    @validator('ACC_x', 'ACC_y', 'ACC_z')
    def validate_array_length(cls, v):
        """Validate that each signal array has exactly 240 samples if provided."""
        if v is not None and len(v) != WINDOW_LEN:
            raise ValueError(f"Array must have exactly {WINDOW_LEN} elements, got {len(v)}")
        return v


class EnvInput(BaseModel):
    """Environmental context inputs."""
    
    temp_c: Optional[float] = Field(None, description="Ambient temperature (°C)")
    humidity: Optional[float] = Field(None, description="Relative humidity (%)")
    aqi: Optional[int] = Field(None, description="Air Quality Index")
    local_hour: Optional[int] = Field(None, ge=0, le=23, description="Local hour [0-23]")
    pressure: Optional[float] = Field(None, description="Atmospheric pressure (hPa)")
    light_level: Optional[float] = Field(None, description="Light level (lux)")
    noise_level: Optional[float] = Field(None, description="Noise level (dB)")


class VisionInput(BaseModel):
    """Vision/visual context inputs."""
    
    embedding: Optional[List[float]] = Field(None, description="Vision embedding vector (e.g., CLIP, ResNet)")
    objects: Optional[List[str]] = Field(None, description="Detected objects")
    scene_type: Optional[str] = Field(None, description="Scene classification (e.g., 'indoor', 'outdoor', 'vehicle')")
    activity_context: Optional[str] = Field(None, description="Activity context (e.g., 'walking', 'sitting', 'operating')")
    
    @validator('embedding')
    def validate_embedding(cls, v):
        """Validate embedding is reasonable size if provided."""
        if v is not None and len(v) > 10000:
            raise ValueError(f"Embedding vector too large: {len(v)} elements (max 10000)")
        return v


class AudioInput(BaseModel):
    """Audio context inputs."""
    
    embedding: Optional[List[float]] = Field(None, description="Audio embedding vector (e.g., Wav2Vec, VGGish)")
    keywords: Optional[List[str]] = Field(None, description="Detected keywords/phrases")
    speech_activity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Speech activity level [0-1]")
    emotion: Optional[str] = Field(None, description="Detected emotion (e.g., 'calm', 'stressed', 'excited')")
    
    @validator('embedding')
    def validate_embedding(cls, v):
        """Validate embedding is reasonable size if provided."""
        if v is not None and len(v) > 10000:
            raise ValueError(f"Embedding vector too large: {len(v)} elements (max 10000)")
        return v


class TaskInput(BaseModel):
    """Task metadata inputs."""
    
    goal: Optional[str] = Field(None, description="Task goal/description")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Task confidence [0-1]")
    priority: Optional[int] = Field(None, ge=0, le=10, description="Task priority [0-10]")
    complexity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Task complexity [0-1]")
    deadline_proximity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Deadline proximity [0-1]")


class SystemInput(BaseModel):
    """System/robotics signals."""
    
    cpu_usage: Optional[float] = Field(None, ge=0.0, le=1.0, description="CPU usage [0-1]")
    memory_usage: Optional[float] = Field(None, ge=0.0, le=1.0, description="Memory usage [0-1]")
    network_latency: Optional[float] = Field(None, ge=0.0, description="Network latency (ms)")
    error_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Error rate [0-1]")
    battery_level: Optional[float] = Field(None, ge=0.0, le=1.0, description="Battery level [0-1]")
    system_load: Optional[float] = Field(None, ge=0.0, description="System load")


class V2CavWindow(BaseModel):
    """Single window for v2 multimodal CAV computation."""
    
    physio: Optional[PhysioInput] = Field(None, description="Physiological signals (EDA, BVP, TEMP, ACC)")
    motion: Optional[MotionInput] = Field(None, description="Motion and torque data (ACC_x/y/z, velocity, torque)")
    env: Optional[EnvInput] = Field(None, description="Environmental context (temp_c, humidity, aqi)")
    vision: Optional[VisionInput] = Field(None, description="Vision/visual context (embeddings, objects)")
    audio: Optional[AudioInput] = Field(None, description="Audio context (embeddings, keywords)")
    task: Optional[TaskInput] = Field(None, description="Task metadata (id, complexity, difficulty)")
    system: Optional[SystemInput] = Field(None, description="System/robotics signals (CPU, memory, battery)")
    device_profile: Optional[str] = Field(None, description="Device profile: humanoid_full, wearable_limited, drone_nav")
    
    @model_validator(mode='after')
    def validate_at_least_one_input(self):
        """Ensure at least one input modality is provided."""
        if not any([
            self.physio, self.motion, self.env,
            self.vision, self.audio, self.task, 
            self.system
        ]):
            raise ValueError("At least one input modality (physio, motion, env, vision, audio, task, system) must be provided")
        return self


# Alias for backward compatibility
CAVRequestV2 = V2CavWindow


class InfluenceFields(BaseModel):
    """Control influence fields for robotics/actuation."""
    
    speed_scale: float = Field(..., ge=0.0, le=2.0, description="Speed scaling factor [0-2]")
    torque_scale: float = Field(..., ge=0.0, le=2.0, description="Torque scaling factor [0-2]")
    safety_scale: float = Field(..., ge=0.0, le=1.0, description="Safety margin scaling [0-1]")
    caution_flag: bool = Field(..., description="Caution flag (true = reduce aggressiveness)")
    emergency_flag: bool = Field(..., description="Emergency flag (true = immediate safety mode)")
    focus_boost: float = Field(..., ge=0.0, le=1.0, description="Focus boost factor [0-1]")
    recovery_recommended: bool = Field(..., description="Recovery recommended flag")


class V2CavResult(BaseModel):
    """Single result for v2 CAV computation."""
    
    ok: bool = Field(True, description="Whether computation succeeded")
    error: Optional[str] = Field(None, description="Error message if ok=false")
    cav_vector: Optional[List[float]] = Field(None, description="Fixed-length CAV embedding vector (128-dim)")
    state_class: Optional[str] = Field(None, description="State: restorative | focus | balanced | alert | overload | emergency")
    p_stress: Optional[float] = Field(None, ge=0.0, le=1.0, description="Probability of stress [0-1]")
    p_chaos: Optional[float] = Field(None, ge=0.0, le=1.0, description="Probability of chaos/overload [0-1]")
    influences: Optional[InfluenceFields] = Field(None, description="Control influence fields")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall confidence in state_class [0-1]")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @model_validator(mode='after')
    def validate_result_fields(self):
        """Ensure required fields are present when ok=true."""
        if self.ok:
            if self.cav_vector is None:
                raise ValueError("cav_vector is required when ok=true")
            if self.state_class is None:
                raise ValueError("state_class is required when ok=true")
            if self.p_stress is None:
                raise ValueError("p_stress is required when ok=true")
            if self.p_chaos is None:
                raise ValueError("p_chaos is required when ok=true")
            if self.influences is None:
                raise ValueError("influences is required when ok=true")
            if self.confidence is None:
                raise ValueError("confidence is required when ok=true")
            if self.metadata is None:
                raise ValueError("metadata is required when ok=true")
        return self


# Alias for backward compatibility
CAVResponseV2 = V2CavResult


class V2CavBatchRequest(BaseModel):
    """Batch request for v2 CAV computation."""
    
    windows: List[V2CavWindow] = Field(..., description="List of multimodal windows to process", min_items=1, max_items=10)


# Alias for backward compatibility
BatchRequestV2 = V2CavBatchRequest


class V2CavBatchResponse(BaseModel):
    """Response model for v2 batch computation."""
    
    results: List[V2CavResult] = Field(..., description="CAV results for each window")
    latency_ms: float = Field(..., description="Total processing latency in milliseconds")
    server_version: str = Field(..., description="Server version string")


# Aliases for backward compatibility
BatchResponseItemV2 = V2CavResult
BatchResponseV2 = V2CavBatchResponse

