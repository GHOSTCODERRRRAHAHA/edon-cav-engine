"""Device profiles for EDON v2 - define sensor availability and weighting."""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum


class DeviceProfile(str, Enum):
    """Device profile types."""
    HUMANOID_FULL = "humanoid_full"
    WEARABLE_LIMITED = "wearable_limited"
    DRONE_NAV = "drone_nav"


@dataclass
class SensorConfig:
    """Configuration for a sensor modality."""
    available: bool
    weight: float
    required: bool = False


@dataclass
class DeviceProfileConfig:
    """Complete device profile configuration."""
    name: str
    description: str
    physio: SensorConfig
    motion: SensorConfig
    env: SensorConfig
    vision: SensorConfig
    audio: SensorConfig
    task: SensorConfig
    system: SensorConfig
    modality_weights: Dict[str, float]
    default_confidence_boost: float = 0.0


# Device Profile Definitions
DEVICE_PROFILES: Dict[DeviceProfile, DeviceProfileConfig] = {
    DeviceProfile.HUMANOID_FULL: DeviceProfileConfig(
        name="humanoid_full",
        description="Full humanoid robot with all sensors",
        # OEM-friendly: no required=True - profile is weighting hint only
        physio=SensorConfig(available=True, weight=1.0, required=False),
        motion=SensorConfig(available=True, weight=1.0, required=False),
        env=SensorConfig(available=True, weight=0.8, required=False),
        vision=SensorConfig(available=True, weight=0.8, required=False),
        audio=SensorConfig(available=True, weight=0.6, required=False),
        task=SensorConfig(available=True, weight=1.0, required=False),
        system=SensorConfig(available=True, weight=0.6, required=False),
        modality_weights={
            'physio': 0.35,
            'motion': 0.20,
            'env': 0.15,
            'vision': 0.15,
            'audio': 0.10,
            'task': 0.03,
            'system': 0.02
        },
        default_confidence_boost=0.1
    ),
    
    DeviceProfile.WEARABLE_LIMITED: DeviceProfileConfig(
        name="wearable_limited",
        description="Wearable device with limited sensors (physio + env only)",
        # OEM-friendly: no required=True - profile is weighting hint only
        physio=SensorConfig(available=True, weight=1.0, required=False),
        motion=SensorConfig(available=True, weight=0.8, required=False),
        env=SensorConfig(available=True, weight=0.8, required=False),
        vision=SensorConfig(available=False, weight=0.0, required=False),
        audio=SensorConfig(available=False, weight=0.0, required=False),
        task=SensorConfig(available=False, weight=0.6, required=False),
        system=SensorConfig(available=False, weight=0.3, required=False),
        modality_weights={
            'physio': 0.70,
            'motion': 0.15,
            'env': 0.10,
            'vision': 0.0,
            'audio': 0.0,
            'task': 0.05,
            'system': 0.0
        },
        default_confidence_boost=0.0
    ),
    
    DeviceProfile.DRONE_NAV: DeviceProfileConfig(
        name="drone_nav",
        description="Autonomous drone navigation stack",
        # OEM-friendly: no required=True - profile is weighting hint only
        physio=SensorConfig(available=False, weight=0.0, required=False),
        motion=SensorConfig(available=True, weight=1.0, required=False),
        env=SensorConfig(available=True, weight=0.7, required=False),
        vision=SensorConfig(available=True, weight=1.0, required=False),
        audio=SensorConfig(available=False, weight=0.3, required=False),
        task=SensorConfig(available=True, weight=0.8, required=False),
        system=SensorConfig(available=True, weight=0.8, required=False),
        modality_weights={
            'physio': 0.0,
            'motion': 0.40,
            'env': 0.10,
            'vision': 0.30,
            'audio': 0.0,
            'task': 0.10,
            'system': 0.10
        },
        default_confidence_boost=0.05
    )
}


def get_profile(profile_name: str) -> Optional[DeviceProfileConfig]:
    """Get device profile by name."""
    try:
        profile_enum = DeviceProfile(profile_name)
        return DEVICE_PROFILES[profile_enum]
    except (ValueError, KeyError):
        return None


def get_available_modalities(profile: DeviceProfileConfig) -> Set[str]:
    """Get set of available modalities for a profile."""
    available = set()
    if profile.physio.available:
        available.add('physio')
    if profile.motion.available:
        available.add('motion')
    if profile.env.available:
        available.add('env')
    if profile.vision.available:
        available.add('vision')
    if profile.audio.available:
        available.add('audio')
    if profile.task.available:
        available.add('task')
    if profile.system.available:
        available.add('system')
    return available


def get_required_modalities(profile: DeviceProfileConfig) -> Set[str]:
    """Get set of required modalities for a profile."""
    required = set()
    if profile.physio.required:
        required.add('physio')
    if profile.motion.required:
        required.add('motion')
    if profile.env.required:
        required.add('env')
    if profile.vision.required:
        required.add('vision')
    if profile.audio.required:
        required.add('audio')
    if profile.task.required:
        required.add('task')
    if profile.system.required:
        required.add('system')
    return required


def validate_request_for_profile(
    request_modalities: Set[str],
    profile: DeviceProfileConfig
) -> Tuple[bool, Optional[str]]:
    """
    DEPRECATED: Device profiles are weighting hints only, not validation contracts.
    
    This function is kept for backwards compatibility but should not be used.
    OEMs can send any combination of modalities regardless of profile.
    
    Returns:
        (is_valid, error_message) - Always returns (True, None) since profiles don't validate
    """
    # Profiles are weighting hints only - never reject requests
    return True, None

