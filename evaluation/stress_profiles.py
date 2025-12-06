"""Stress profile definitions for environment difficulty levels."""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class StressProfile:
    """Configuration for a stress profile."""
    
    name: str
    push_force_min: float  # N
    push_force_max: float  # N
    push_probability: float  # per step
    sensor_noise_std: float
    friction_min: float
    friction_max: float
    actuator_delay_steps: tuple[int, int]  # (min, max) steps
    fatigue_enabled: bool
    fatigue_degradation: float  # Performance degradation over episode (0.0-1.0)
    floor_incline_range: tuple[float, float]  # (min, max) radians
    height_variation_range: tuple[float, float]  # (min, max) meters
    description: str = ""


# Define stress profiles
LIGHT_STRESS = StressProfile(
    name="light_stress",
    push_force_min=5.0,
    push_force_max=50.0,
    push_probability=0.08,
    sensor_noise_std=0.01,
    friction_min=0.5,
    friction_max=1.0,
    actuator_delay_steps=(0, 0),  # No delay
    fatigue_enabled=False,
    fatigue_degradation=0.0,
    floor_incline_range=(0.0, 0.0),  # No incline
    height_variation_range=(0.0, 0.0),  # No height variation
    description="Light stress: minimal disturbances, low noise"
)

MEDIUM_STRESS = StressProfile(
    name="medium_stress",
    push_force_min=10.0,
    push_force_max=100.0,
    push_probability=0.12,
    sensor_noise_std=0.02,
    friction_min=0.3,
    friction_max=1.2,
    actuator_delay_steps=(1, 2),  # 10-20ms delay
    fatigue_enabled=True,
    fatigue_degradation=0.05,  # 5% degradation
    floor_incline_range=(-0.05, 0.05),  # Small incline
    height_variation_range=(0.0, 0.0),
    description="Medium stress: moderate disturbances, some delays"
)

HIGH_STRESS = StressProfile(
    name="high_stress",
    push_force_min=20.0,
    push_force_max=150.0,
    push_probability=0.18,
    sensor_noise_std=0.03,
    friction_min=0.2,
    friction_max=1.5,
    actuator_delay_steps=(2, 4),  # 20-40ms delay
    fatigue_enabled=True,
    fatigue_degradation=0.10,  # 10% degradation
    floor_incline_range=(-0.15, 0.15),  # ±8.6 degrees
    height_variation_range=(-0.05, 0.05),  # ±5cm
    description="High stress: strong disturbances, delays, fatigue, uneven terrain"
)

HELL_STRESS = StressProfile(
    name="hell_stress",
    push_force_min=120.0,  # Much higher minimum force
    push_force_max=220.0,  # Extreme maximum force
    push_probability=0.65,  # Very frequent pushes (65% per step)
    sensor_noise_std=0.04,  # High sensor noise
    friction_min=0.3,  # Very low friction possible
    friction_max=1.2,  # High friction possible
    actuator_delay_steps=(3, 6),  # 30-60ms delay (extreme)
    fatigue_enabled=True,
    fatigue_degradation=0.20,  # 20% degradation (severe fatigue)
    floor_incline_range=(-0.12, 0.12),  # ±6.9 degrees (unstable terrain)
    height_variation_range=(-0.08, 0.08),  # ±8cm (significant height variation)
    description="Hell mode: extreme disturbances, heavy delays, unstable terrain"
)

# Profile registry
STRESS_PROFILES: Dict[str, StressProfile] = {
    "light_stress": LIGHT_STRESS,
    "medium_stress": MEDIUM_STRESS,
    "high_stress": HIGH_STRESS,
    "hell_stress": HELL_STRESS,
}


def get_stress_profile(name: str) -> StressProfile:
    """Get stress profile by name."""
    if name not in STRESS_PROFILES:
        raise ValueError(f"Unknown stress profile: {name}. Available: {list(STRESS_PROFILES.keys())}")
    return STRESS_PROFILES[name]


def list_profiles() -> list[str]:
    """List available stress profile names."""
    return list(STRESS_PROFILES.keys())

