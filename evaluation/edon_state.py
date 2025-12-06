"""EDON state classification and mapping utilities."""

from enum import Enum
from typing import Dict, Any, Optional


class EdonState(Enum):
    """EDON internal state classifications."""
    BALANCED = 0
    FOCUS = 1
    STRESS = 2  # Maps to "alert" state
    OVERLOAD = 3
    RESTORATIVE = 4
    EMERGENCY = 5  # Maps to "emergency" state


def map_edon_output_to_state(edon_output: Optional[Dict[str, Any]]) -> EdonState:
    """
    Map raw EDON output to clean EdonState enum.
    
    Uses p_stress and p_chaos as PRIMARY indicators (continuous metrics).
    Only uses state_class as a secondary check if p_stress/p_chaos are missing.
    
    Args:
        edon_output: Dictionary from EDON client with:
            - state_class: str ("restorative", "focus", "balanced", "alert", "overload", "emergency")
            - p_stress: float [0-1] - PRIMARY indicator
            - p_chaos: float [0-1] - PRIMARY indicator
            - influences: dict with emergency_flag, caution_flag, etc.
    
    Returns:
        EdonState enum value
    """
    if edon_output is None:
        return EdonState.BALANCED  # Default if EDON unavailable
    
    # Check for emergency first (highest priority)
    influences = edon_output.get("influences", {})
    if isinstance(influences, dict) and influences.get("emergency_flag", False):
        return EdonState.EMERGENCY
    
    # PRIMARY: Use p_stress and p_chaos (continuous metrics) to infer state
    # These are more reliable than state_class which might be trivial/constant
    p_stress = edon_output.get("p_stress", 0.0)
    p_chaos = edon_output.get("p_chaos", 0.0)
    
    # Ensure we have valid numeric values
    if not isinstance(p_stress, (int, float)) or not isinstance(p_chaos, (int, float)):
        p_stress = 0.0
        p_chaos = 0.0
    
    # Use thresholds to map continuous metrics to discrete states
    # OVERLOAD: High chaos (system overwhelmed)
    if p_chaos > 0.65:
        return EdonState.OVERLOAD
    
    # STRESS: High stress but manageable chaos
    elif p_stress > 0.55:
        return EdonState.STRESS
    
    # RESTORATIVE: Low stress and low chaos (recovery mode)
    elif p_stress < 0.25 and p_chaos < 0.15:
        return EdonState.RESTORATIVE
    
    # FOCUS: Moderate stress, low chaos (optimal performance)
    elif p_stress < 0.45 and p_chaos < 0.25:
        return EdonState.FOCUS
    
    # BALANCED: Default for moderate stress/chaos
    else:
        # Secondary check: use state_class if it's meaningful
        state_class = edon_output.get("state_class", "")
        if isinstance(state_class, str):
            state_class = state_class.lower()
            if state_class in ("overload", "alert", "emergency"):
                # If state_class suggests stress/overload but p_stress/p_chaos don't, trust state_class
                if state_class == "overload":
                    return EdonState.OVERLOAD
                elif state_class in ("alert", "emergency"):
                    return EdonState.STRESS
        
        return EdonState.BALANCED


def compute_risk_score(edon_output: Optional[Dict[str, Any]]) -> float:
    """
    Compute continuous risk score from EDON's p_stress and p_chaos.
    
    Risk score indicates how urgently EDON should act to stabilize.
    Higher risk = more aggressive corrections needed.
    
    Args:
        edon_output: Dictionary from EDON client with p_stress, p_chaos
    
    Returns:
        Risk score in [0, 1] where:
        - 0.0 = very low risk (restorative)
        - 0.5 = moderate risk (balanced/stress)
        - 1.0 = very high risk (overload/emergency)
    """
    if edon_output is None:
        return 0.0  # No risk if EDON unavailable
    
    p_stress = edon_output.get("p_stress", 0.0)
    p_chaos = edon_output.get("p_chaos", 0.0)
    
    # Ensure valid numeric values
    if not isinstance(p_stress, (int, float)) or not isinstance(p_chaos, (int, float)):
        return 0.0
    
    # Risk = weighted combination of chaos (more critical) and stress
    # Chaos is weighted more heavily as it indicates system overwhelm
    risk = 0.6 * p_chaos + 0.4 * p_stress
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, risk))

