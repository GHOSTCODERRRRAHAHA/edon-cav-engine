"""State classification for EDON v2 with extended states."""

from typing import Dict, Any, Tuple
import numpy as np


def classify_state_v2(
    p_stress: float,
    p_focus: float,
    p_chaos: float,
    env_score: float = 0.5,
    circadian_score: float = 0.5,
    system_stress: float = 0.0,
    emergency_indicators: Dict[str, Any] = None
) -> str:
    """
    Classify state into one of: restorative | focus | balanced | overload | alert | emergency
    
    Demo-friendly classification logic that is more sensitive to changes in p_stress and p_chaos.
    Conditions are checked in order from restorative → emergency.
    
    Args:
        p_stress: Probability of stress [0-1]
        p_focus: Probability of focus [0-1]
        p_chaos: Probability of chaos/overload [0-1]
        env_score: Environmental comfort score [0-1]
        circadian_score: Circadian alignment score [0-1]
        system_stress: System stress indicator [0-1]
        emergency_indicators: Dict with emergency flags (e.g., {'has_emergency': True})
    
    Returns:
        State class string
    """
    if emergency_indicators is None:
        emergency_indicators = {}
    
    # State classification: check in order from most severe to least severe
    # Order: emergency > overload > alert > balanced > focus > restorative
    
    # 1. Emergency: explicit flags OR extremely high stress + chaos
    if emergency_indicators.get('has_emergency', False):
        return 'emergency'
    
    if emergency_indicators.get('system_critical', False):
        return 'emergency'
    
    if system_stress > 0.9:
        return 'emergency'
    
    # 2. Emergency: extremely high stress and chaos
    if p_stress >= 0.90 and p_chaos >= 0.75:
        return 'emergency'
    
    # 3. Overload: very high stress, high chaos
    if p_stress >= 0.80 and p_chaos >= 0.60:
        return 'overload'
    
    # 4. Alert: high stress, moderate chaos
    if p_stress >= 0.70 and p_chaos < 0.60:
        return 'alert'
    
    # 5. Focus: moderate stress, low chaos (requires specific band)
    if 0.20 <= p_stress <= 0.45 and p_chaos < 0.20:
        return 'focus'
    
    # 6. Restorative: very low stress and chaos
    if p_stress < 0.20 and p_chaos < 0.15:
        return 'restorative'
    
    # 7. Balanced: default for moderate stress/chaos (not in other bands)
    # This catches: p_stress < 0.70 and p_chaos < 0.40 but not focus/restorative
    return 'balanced'


def compute_influence_fields(
    state: str,
    p_stress: float,
    p_focus: float,
    p_chaos: float,
    env_score: float = 0.5,
    system_stress: float = 0.0
) -> Dict[str, Any]:
    """
    Compute control influence fields based on state and probabilities.
    
    Returns:
        Dictionary with speed_scale, torque_scale, safety_scale, flags, etc.
    """
    influences = {
        'speed_scale': 1.0,
        'torque_scale': 1.0,
        'safety_scale': 0.85,
        'caution_flag': False,
        'emergency_flag': False,
        'focus_boost': 0.0,
        'recovery_recommended': False
    }
    
    if state == 'emergency':
        influences['speed_scale'] = 0.1
        influences['torque_scale'] = 0.1
        influences['safety_scale'] = 1.0
        influences['caution_flag'] = True
        influences['emergency_flag'] = True
        influences['recovery_recommended'] = True
    
    elif state == 'alert':
        influences['speed_scale'] = 0.3
        influences['torque_scale'] = 0.3
        influences['safety_scale'] = 0.95
        influences['caution_flag'] = True
        influences['recovery_recommended'] = True
    
    elif state == 'overload':
        influences['speed_scale'] = 0.4
        influences['torque_scale'] = 0.4
        influences['safety_scale'] = 1.0
        influences['caution_flag'] = True
        influences['recovery_recommended'] = True
    
    elif state == 'restorative':
        influences['speed_scale'] = 0.7
        influences['torque_scale'] = 0.7
        influences['safety_scale'] = 0.95
        influences['recovery_recommended'] = False
    
    elif state == 'focus':
        influences['speed_scale'] = 1.2
        influences['torque_scale'] = 1.1
        influences['safety_scale'] = 0.8
        influences['focus_boost'] = min(p_focus, 0.3)  # Boost up to 30%
        influences['caution_flag'] = False
    
    else:  # balanced
        influences['speed_scale'] = 1.0
        influences['torque_scale'] = 1.0
        influences['safety_scale'] = 0.85
        influences['caution_flag'] = p_stress > 0.5
    
    # Adjust based on system stress
    if system_stress > 0.5:
        influences['speed_scale'] *= (1.0 - system_stress * 0.3)
        influences['torque_scale'] *= (1.0 - system_stress * 0.3)
        influences['safety_scale'] = min(1.0, influences['safety_scale'] + system_stress * 0.1)
    
    # Clamp values
    influences['speed_scale'] = max(0.0, min(2.0, influences['speed_scale']))
    influences['torque_scale'] = max(0.0, min(2.0, influences['torque_scale']))
    influences['safety_scale'] = max(0.0, min(1.0, influences['safety_scale']))
    influences['focus_boost'] = max(0.0, min(1.0, influences['focus_boost']))
    
    return influences

