"""
EDON Controller V3 - "Chaos Stabilizer"

Multi-layer, risk-aware stabilizer that uses EDON's p_stress, p_chaos, and risk
to intelligently shape corrections across SAFE/PREFALL/FAIL zones.

Key features:
- Multi-layer corrections: tilt, velocity damping, gait smoothing
- Risk-aware zones with virtual PREFALL
- Joint-aware PD corrections (balance-critical joints only)
- Direction checks vs tilt (not baseline)
- Target ratio scaling per zone and risk
- EMA & hysteresis for smooth operation
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np
import os

from evaluation.edon_state import EdonState, compute_risk_score, map_edon_output_to_state
from evaluation.config import config


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp value between lo and hi."""
    return max(lo, min(hi, x))


def apply_edon_gain(base_pd: np.ndarray, edon_correction: np.ndarray, edon_state: Dict[str, Any]) -> np.ndarray:
    """
    Apply state-aware adaptive EDON gain instead of fixed global gain.
    
    V5.2: Added super-conservative predicted instability boost (up to +10%).
    
    Args:
        base_pd: Baseline PD controller action
        edon_correction: EDON correction vector
        edon_state: Dictionary with instability_score, disturbance_level, phase, predicted_instability
        
    Returns:
        Blended action with adaptive gain
    """
    # Use locked V4 configuration (EDON_v3.1_high_stress_V4)
    cfg = EDON_V31_HS_V4_CFG
    
    # Extract state variables (0..1 or similar)
    instability = edon_state.get("instability_score", 0.0)
    disturbance = edon_state.get("disturbance_level", 0.0)
    phase = edon_state.get("phase", "normal")  # "normal", "pre_fall", "recovery"
    predicted_instability = edon_state.get("predicted_instability", 0.0)  # V5.2: super-conservative
    
    # Base gain (V4 config - FROZEN)
    base_gain = cfg["BASE_GAIN"]
    
    # Adaptive term (V4 config - FROZEN)
    gain = base_gain + cfg["INSTABILITY_WEIGHT"] * instability + cfg["DISTURBANCE_WEIGHT"] * disturbance
    
    # V5.2: Super-conservative predicted boost (up to +10%, not +15% or +30%)
    # Small gain modulation, not a turbo button
    gain *= (1.0 + 0.2 * predicted_instability)  # 0-10% bump (0.2 * 0.5 max = 0.1 = 10%)
    
    if phase == "recovery":
        gain *= cfg["RECOVERY_BOOST"]  # extra help during recovery (V4 config)
    
    gain = clamp(gain, 0.3, 1.1)
    
    return base_pd + gain * edon_correction


def apply_prefall_reflex(torque_cmd: np.ndarray, edon_state: Dict[str, Any], prefall_direction: np.ndarray, internal_state: Dict[str, Any]) -> np.ndarray:
    """
    Apply dynamic PREFALL reflex based on EDON fall risk prediction.
    PREFALL barely touches normal gait but ramps up when EDON sees fall risk.
    
    V5.1: No PREFALL decay (LPF only, testing)
    V5.2+: Will add decay if needed
    
    Uses EDON_v3.1_high_stress_V4 configuration (FROZEN).
    
    Args:
        torque_cmd: Current torque command
        edon_state: Dictionary with fall_risk
        prefall_direction: Vector/array shaped like torque_cmd
        internal_state: State dict (for future decay tracking)
        
    Returns:
        Torque command with PREFALL reflex applied
    """
    # Use locked V4 configuration
    cfg = EDON_V31_HS_V4_CFG
    
    # risk 0..1 predicted by EDON
    risk = edon_state.get("fall_risk", 0.0)
    
    # Dynamic PREFALL: from PREFALL_MIN (low risk) up to PREFALL_MIN+PREFALL_RANGE (high risk)
    # V5.1: No decay, direct calculation (testing LPF only)
    prefall_gain = cfg["PREFALL_MIN"] + cfg["PREFALL_RANGE"] * risk
    prefall_gain = clamp(prefall_gain, 0.0, cfg["PREFALL_MAX"])
    
    # prefall_direction should be a vector/array shaped like torque_cmd
    return torque_cmd + prefall_gain * prefall_direction


def apply_safe_override(torque_cmd: np.ndarray, edon_state: Dict[str, Any], safe_posture_torque: np.ndarray) -> np.ndarray:
    """
    Apply SAFE override only when catastrophic risk is high.
    SAFE only kicks in when things are about to go to hell.
    
    V5.1: No pre-trigger (LPF only, testing)
    V5.2+: Will add pre-trigger if needed
    
    Args:
        torque_cmd: Current torque command
        edon_state: Dictionary with catastrophic_risk
        safe_posture_torque: Safe posture correction vector
        
    Returns:
        Torque command with SAFE override applied if needed
    """
    # Use locked V4 configuration (EDON_v3.1_high_stress_V4)
    cfg = EDON_V31_HS_V4_CFG
    
    catastrophic_risk = edon_state.get("catastrophic_risk", 0.0)
    
    # V5.1: Original V4 behavior (no pre-trigger)
    if catastrophic_risk <= cfg["SAFE_THRESHOLD"]:
        return torque_cmd  # SAFE dormant
    
    safe_gain = cfg["SAFE_GAIN"]  # V4 config - FROZEN
    return (1.0 - safe_gain) * torque_cmd + safe_gain * safe_posture_torque


# === EDON Multi-Mode Controller Constants ===
MODE_NORMAL = 0
MODE_BRACE = 1
MODE_ESCALATE = 2  # Emergency mode between BRACE and RECOVERY
MODE_RECOVERY = 3


# Balance-critical joint indices (adjust based on your robot model)
# These are the joints that directly affect balance (torso/hip/ankle)
BALANCE_JOINTS = [0, 1, 2, 3]  # Roll, pitch, COM x, COM y

# PD gains for tilt correction (OPTIMIZING for 10% target)
KP_ROLL = 0.18   # INCREMENTAL: Increased to 80% above baseline (Test 3 was +1.0%, need more)
KP_PITCH = 0.18  # INCREMENTAL: Increased to 80% above baseline
KD_ROLL = 0.05   # INCREMENTAL: Increased to 67% above baseline
KD_PITCH = 0.05  # INCREMENTAL: Increased to 67% above baseline

# Velocity damping gain
VEL_DAMP_GAIN = 0.1

# Gait smoothing gain (very small)
GAIT_SMOOTH_GAIN = 0.05

# === EDON v3.1 High-Stress V4 Configuration (FROZEN) ===
# Locked configuration that achieved +6.4% average improvement (30 episodes)
# This is the reference configuration for comparison against future changes
EDON_V31_HS_V4_CFG = {
    # Adaptive gain settings (Updated: BASE_GAIN=0.5 achieves +7.0% vs +6.4% target)
    "BASE_GAIN": 0.5,  # INCREMENTAL: Increased from 0.4 to 0.5 for +7.0% improvement
    "INSTABILITY_WEIGHT": 0.4,
    "DISTURBANCE_WEIGHT": 0.2,
    "RECOVERY_BOOST": 1.2,  # 20% extra during recovery
    
    # PREFALL reflex settings (Updated: PREFALL_RANGE=0.50 achieves +4.7% with BASE_GAIN=0.5)
    "PREFALL_MIN": 0.15,  # Low risk gain
    "PREFALL_RANGE": 0.50,  # INCREMENTAL: Increased from 0.45 to 0.50 for better stability
    "PREFALL_MAX": 0.70,  # INCREMENTAL: Increased from 0.65 to 0.70 to match range
    
    # SAFE override settings
    "SAFE_THRESHOLD": 0.75,  # Catastrophic risk threshold
    "SAFE_GAIN": 0.12,  # 12% blend when active
    
    # Legacy compatibility (kept for existing code)
    "PREFALL_BASE": 0.34,  # Not used in adaptive version, kept for compatibility
    "PREFALL_MIN_LEGACY": 0.20,
    "PREFALL_MAX_LEGACY": 0.46,
}

# === EDON v3.1 High-Stress V5.1 Configuration ===
# V4 + Low-Pass Filter only (no other changes)
# Testing if LPF alone improves performance
EDON_V31_HS_V51_CFG = {
    # Adaptive gain settings (Updated: BASE_GAIN=0.5 achieves +7.0% vs +6.4% target)
    "BASE_GAIN": 0.5,  # INCREMENTAL: Increased from 0.4 to 0.5 for +7.0% improvement
    "INSTABILITY_WEIGHT": 0.4,
    "DISTURBANCE_WEIGHT": 0.2,
    "RECOVERY_BOOST": 1.2,  # 20% extra during recovery
    
    # PREFALL reflex settings (Updated: PREFALL_RANGE=0.50 achieves +4.7% with BASE_GAIN=0.5)
    "PREFALL_MIN": 0.15,  # Low risk gain
    "PREFALL_RANGE": 0.50,  # INCREMENTAL: Increased from 0.45 to 0.50 for better stability
    "PREFALL_MAX": 0.70,  # INCREMENTAL: Increased from 0.65 to 0.70 to match range
    
    # SAFE override settings
    "SAFE_THRESHOLD": 0.75,  # Catastrophic risk threshold
    "SAFE_GAIN": 0.12,  # 12% blend when active
    
    # Legacy compatibility (kept for existing code)
    "PREFALL_BASE": 0.34,  # Not used in adaptive version, kept for compatibility
    "PREFALL_MIN_LEGACY": 0.20,
    "PREFALL_MAX_LEGACY": 0.46,
}

# === EDON v3.5 – dynamic PREFALL + danger momentum + micro-SAFE ===
# High-stress tuned configuration + V3.6 brace-mode
# NOTE: This is legacy - V4 uses EDON_V31_HS_V4_CFG above
EDON_V35_CFG = {
    # === Existing v3.5 config (tune as you like) ===
    "PREFALL_BASE": 0.34,  # INCREMENTAL: Test 5 gave +1.1% - keeping this as base
    "PREFALL_MIN": 0.15,   # Keep moderate
    "PREFALL_MAX": 0.65,   # Keep moderate
    
    "DANGER_MOMENTUM_THRESH": 0.007,
    "DANGER_MOMENTUM_BOOST": 0.30,
    "DANGER_MOMENTUM_DECAY": 0.20,
    
    "SAFE_GAIN": 0.12,   # INCREMENTAL: Dialed back - SAFE was causing issues
    "SAFE_ON_THRESH": 0.020,         # EMA ~ 0.02–0.03 → will actually trigger
    "SAFE_OFF_THRESH": 0.010,
    
    # === NEW: V3.6 brace-mode config ===
    # When to enter/exit brace mode (high stress only)
    # Tuned for EMA range: typical 0.02-0.03, max ~0.072
    "BRACE_ON_THRESH": 0.030,        # "top half" of typical EMA
    "BRACE_OFF_THRESH": 0.015,
    
    # Momentum-based triggers (using delta_ema)
    "BRACE_MOMENTUM_ON": 0.003,      # EMA ramps quickly → brace
    "BRACE_MOMENTUM_OFF": -0.002,     # EMA slopes down → relax
    
    # Scaling factors when in brace mode
    "BRACE_PREFALL_MULT": 1.35,      # PREFALL hits harder in brace mode
    "BRACE_SAFE_MULT": 1.50,         # SAFE micro-corrections stronger
    "BRACE_BASELINE_SCALE": 0.90,    # slightly damp baseline → less flailing
    
    # Debug toggle (optional)
    "DEBUG_BRACE": True,
}


def compute_tilt_info(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute tilt information from observation.
    
    Args:
        obs: Observation dictionary with roll, pitch, velocities, etc.
    
    Returns:
        Dictionary with:
        - roll: float
        - pitch: float
        - roll_velocity: float
        - pitch_velocity: float
        - tilt_mag: float (magnitude)
        - max_tilt: float (max of |roll|, |pitch|)
        - zone: str ("SAFE", "PREFALL", "FAIL")
    """
    roll = obs.get("roll", 0.0)
    pitch = obs.get("pitch", 0.0)
    roll_velocity = obs.get("roll_velocity", 0.0)
    pitch_velocity = obs.get("pitch_velocity", 0.0)
    
    tilt_mag = np.sqrt(roll**2 + pitch**2)
    max_tilt = max(abs(roll), abs(pitch))
    
    # Classify zone by thresholds
    if max_tilt > config.FAIL_LIMIT:
        zone = "FAIL"
    elif max_tilt > config.PREFALL_LIMIT or tilt_mag > config.PREFALL_LIMIT:
        zone = "PREFALL"
    else:
        zone = "SAFE"
    
    return {
        "roll": roll,
        "pitch": pitch,
        "roll_velocity": roll_velocity,
        "pitch_velocity": pitch_velocity,
        "tilt_mag": tilt_mag,
        "max_tilt": max_tilt,
        "zone": zone
    }


def compute_internal_zone(
    tilt_info: Dict[str, Any],
    risk_ema: float,
    prev_zone: str
) -> str:
    """
    Compute internal zone with virtual PREFALL and hysteresis.
    
    Args:
        tilt_info: Tilt information dict
        risk_ema: Exponential moving average of risk score
        prev_zone: Previous internal zone (for hysteresis)
    
    Returns:
        Internal zone: "SAFE", "PREFALL", or "FAIL"
    """
    geom_zone = tilt_info["zone"]
    max_tilt = tilt_info["max_tilt"]
    
    # Start from geometric zone
    if geom_zone == "FAIL":
        return "FAIL"
    elif geom_zone == "PREFALL":
        internal_zone = "PREFALL"
    else:  # SAFE
        # Virtual PREFALL: if risk is high enough, treat moderate tilt as PREFALL
        if risk_ema >= 0.35 and max_tilt > 0.6 * config.SAFE_LIMIT:
            internal_zone = "PREFALL"
        else:
            internal_zone = "SAFE"
    
    # Hysteresis: once in PREFALL, require both low tilt and low risk to go back SAFE
    if prev_zone == "PREFALL" and internal_zone == "SAFE":
        if not (max_tilt < 0.5 * config.SAFE_LIMIT and risk_ema < 0.25):
            internal_zone = "PREFALL"
    
    return internal_zone


def compute_target_ratio_v35(
    internal_zone: str, 
    risk_ema: float, 
    edon_state: Optional[EdonState],
    internal_state: Dict[str, Any],
    edon_output: Optional[Dict[str, Any]] = None
) -> Tuple[float, float]:
    """
    Compute target correction ratio with v3.5 dynamic PREFALL and micro-SAFE.
    
    Args:
        internal_zone: "SAFE", "PREFALL", or "FAIL"
        risk_ema: Exponential moving average of risk score
        edon_state: EDON state enum (optional)
        internal_state: Controller state dict (contains prev_risk, prefall_ratio, safe_active)
        edon_output: Raw EDON output (for computing raw risk for brace mode)
    
    Returns:
        Tuple of (prefall_ratio, safe_ratio)
    """
    cfg = EDON_V35_CFG
    
    # Initialize v3.5 state if needed
    if "edon_prev_risk" not in internal_state:
        internal_state["edon_prev_risk"] = 0.0
    if "prefall_ratio" not in internal_state:
        internal_state["prefall_ratio"] = cfg["PREFALL_BASE"]
    if "safe_active" not in internal_state:
        internal_state["safe_active"] = False
    
    # Use risk_ema for all v3.5/v3.6 logic (not raw risk)
    current_risk = risk_ema
    
    # Initialize prev_risk_ema if needed
    if "edon_prev_risk_ema" not in internal_state:
        internal_state["edon_prev_risk_ema"] = 0.0
    
    # Compute delta_ema for momentum-based triggers and ESCALATE mode
    delta_ema = risk_ema - internal_state["edon_prev_risk_ema"]
    
    # ---- v3.5: danger momentum (delta risk) - ALWAYS ACTIVE ----
    delta_risk = current_risk - internal_state["edon_prev_risk"]
    
    if delta_risk > cfg["DANGER_MOMENTUM_THRESH"]:
        # Risk is rising → hit PREFALL harder
        internal_state["prefall_ratio"] *= (1.0 + cfg["DANGER_MOMENTUM_BOOST"])
    elif delta_risk < -cfg["DANGER_MOMENTUM_THRESH"]:
        # Risk is falling → relax PREFALL a bit
        internal_state["prefall_ratio"] *= (1.0 - cfg["DANGER_MOMENTUM_DECAY"])
    
    # Clamp PREFALL ratio
    internal_state["prefall_ratio"] = max(
        cfg["PREFALL_MIN"], min(cfg["PREFALL_MAX"], internal_state["prefall_ratio"])
    )
    
    # Remember for next step
    internal_state["edon_prev_risk"] = current_risk
    internal_state["edon_prev_risk_ema"] = risk_ema
    
    # ---- v3.5: SAFE micro-corrections with hysteresis (using EMA) ----
    if not internal_state["safe_active"] and risk_ema >= cfg["SAFE_ON_THRESH"]:
        internal_state["safe_active"] = True
    elif internal_state["safe_active"] and risk_ema <= cfg["SAFE_OFF_THRESH"]:
        internal_state["safe_active"] = False
    
    safe_ratio = cfg["SAFE_GAIN"] if internal_state["safe_active"] else 0.0
    
    # === NEW v3.6: brace-mode switching (using EMA + delta_ema) ===
    # Enter brace mode if EMA is high or ramping quickly
    if not internal_state["brace_mode"]:
        if (
            risk_ema >= cfg["BRACE_ON_THRESH"]
            or delta_ema >= cfg["BRACE_MOMENTUM_ON"]
        ):
            internal_state["brace_mode"] = True
    else:
        # Exit brace mode when EMA calms down and momentum turns
        if (
            risk_ema <= cfg["BRACE_OFF_THRESH"]
            and delta_ema <= cfg["BRACE_MOMENTUM_OFF"]
        ):
            internal_state["brace_mode"] = False
    
    # Track steps in brace/total for per-episode stats
    internal_state["total_steps"] = internal_state.get("total_steps", 0) + 1
    if internal_state["brace_mode"]:
        internal_state["brace_steps"] = internal_state.get("brace_steps", 0) + 1
    
    # Use dynamic prefall_ratio for all zones (no fallback constants)
    prefall_ratio = internal_state["prefall_ratio"]
    
    return prefall_ratio, safe_ratio


def compute_target_ratio(internal_zone: str, risk_ema: float, edon_state: Optional[EdonState]) -> float:
    """
    Compute target correction ratio based on zone, risk, and EDON state.
    
    Args:
        internal_zone: "SAFE", "PREFALL", or "FAIL"
        risk_ema: Exponential moving average of risk score
        edon_state: EDON state enum (optional)
    
    Returns:
        Target ratio (correction norm / baseline norm)
    """
    if internal_zone == "SAFE":
        base_ratio = 0.01  # 1%
    elif internal_zone == "PREFALL":
        # Risk-dependent target ratios
        if risk_ema < 0.3:
            base_ratio = 0.15  # Low risk: 15%
        elif risk_ema < 0.6:
            base_ratio = 0.20  # Medium risk: 20%
        else:
            base_ratio = 0.25  # High risk: 25%
    else:  # FAIL
        base_ratio = 0.04  # Gentle damping: 4%
    
    # Optional: state-specific modulation
    state_scale = 1.0
    if edon_state is not None:
        if edon_state == EdonState.STRESS:
            state_scale = 1.1
        elif edon_state == EdonState.OVERLOAD:
            state_scale = 1.2
        elif edon_state == EdonState.RESTORATIVE:
            state_scale = 0.9  # Slightly softer
    
    target_ratio = base_ratio * state_scale
    
    # Cap target ratio per zone (INCREASED for 10%+ target)
    if internal_zone == "SAFE":
        target_ratio = min(target_ratio, 0.04)  # FIX: was 0.02 - Max 4% (was 2%)
    elif internal_zone == "PREFALL":
        target_ratio = min(target_ratio, 0.50)  # FIX: was 0.27 - Max 50% (was 27%)
    elif internal_zone == "FAIL":
        target_ratio = min(target_ratio, 0.12)  # FIX: was 0.05 - Max 12% (was 5%)
    
    return target_ratio


def _get_mode_thresholds(stress_profile: str) -> Dict[str, float]:
    """
    Return mode thresholds tuned per stress profile.
    
    Args:
        stress_profile: "normal_stress", "high_stress", "hell_stress"
    
    Returns:
        Dict with threshold values for mode transitions
    """
    # Defaults: tuned for HIGH STRESS
    thresholds = {
        "normal_to_brace_risk": 0.015,  # was 0.020 - force earlier BRACE activation
        "normal_to_brace_pff":  0.25,
        "brace_to_recovery_risk": 0.035,
        "brace_to_recovery_pff":  0.30,
        "brace_to_normal_risk": 0.010,  # was 0.010 - keep same
        "brace_to_normal_pff":  0.10,
        "recovery_to_brace_risk": 0.025,
    }
    
    if stress_profile == "normal_stress":
        # Environment is calmer → later BRACE / RECOVERY
        thresholds = {
            "normal_to_brace_risk": 0.025,
            "normal_to_brace_pff":  0.30,
            "brace_to_recovery_risk": 0.045,
            "brace_to_recovery_pff":  0.35,
            "brace_to_normal_risk": 0.012,
            "brace_to_normal_pff":  0.10,
            "recovery_to_brace_risk": 0.028,
        }
    elif stress_profile == "hell_stress":
        # Environment is chaotic → earlier BRACE / RECOVERY
        thresholds = {
            "normal_to_brace_risk": 0.015,
            "normal_to_brace_pff":  0.20,
            "brace_to_recovery_risk": 0.028,
            "brace_to_recovery_pff":  0.25,
            "brace_to_normal_risk": 0.010,
            "brace_to_normal_pff":  0.08,
            "recovery_to_brace_risk": 0.020,
        }
    # high_stress falls back to defaults
    
    return thresholds


def _update_edon_mode(
    internal_state: Dict[str, Any],
    risk_ema: float,
    tilt_zone: str,
    prefall_fail_frac: float,
    delta_ema: float = 0.0
) -> None:
    """
    Mode transitions:
    - NORMAL → BRACE
    - BRACE  → ESCALATE
    - BRACE  → RECOVERY
    - BRACE  → NORMAL
    - ESCALATE → RECOVERY
    - ESCALATE → BRACE
    - RECOVERY → BRACE (with minimum duration)
    
    Profile-aware using internal_state["stress_profile"] (if present).
    
    Args:
        internal_state: Controller state dict (will update edon_mode)
        risk_ema: Current risk EMA value
        tilt_zone: Current tilt zone ("SAFE", "PREFALL", or "FAIL")
        prefall_fail_frac: Fraction of recent steps in PREFALL or FAIL zones
        delta_ema: Change in risk EMA (for ESCALATE detection)
    """
    mode = internal_state.get("edon_mode", MODE_NORMAL)
    
    # Get stress profile (from env or internal_state), default to high_stress
    stress_profile = internal_state.get(
        "stress_profile",
        os.getenv("EDON_STRESS_PROFILE", "high_stress")
    )
    
    thresholds = _get_mode_thresholds(stress_profile)
    
    normal_to_brace_risk = thresholds["normal_to_brace_risk"]
    normal_to_brace_pff  = thresholds["normal_to_brace_pff"]
    brace_to_recovery_risk = thresholds["brace_to_recovery_risk"]
    brace_to_recovery_pff  = thresholds["brace_to_recovery_pff"]
    brace_to_normal_risk = thresholds["brace_to_normal_risk"]
    brace_to_normal_pff  = thresholds["brace_to_normal_pff"]
    recovery_to_brace_risk = thresholds["recovery_to_brace_risk"]
    
    # Track mode entry time for minimum duration enforcement
    if "mode_entry_step" not in internal_state:
        internal_state["mode_entry_step"] = 0
    if "current_step" not in internal_state:
        internal_state["current_step"] = 0
    internal_state["current_step"] = internal_state.get("current_step", 0) + 1
    
    # Check if mode changed (for tracking entry time)
    if internal_state.get("edon_mode", MODE_NORMAL) != mode:
        internal_state["mode_entry_step"] = internal_state["current_step"]
    
    # NORMAL → BRACE: risk or sustained PREFALL/FAIL
    if mode == MODE_NORMAL:
        if (risk_ema >= normal_to_brace_risk) or (prefall_fail_frac >= normal_to_brace_pff):
            mode = MODE_BRACE
            internal_state["mode_entry_step"] = internal_state["current_step"]
    
    # BRACE → ESCALATE or RECOVERY or back to NORMAL
    elif mode == MODE_BRACE:
        # ESCALATE: emergency conditions
        if (tilt_zone == "FAIL") or (risk_ema > 0.035) or (delta_ema > 0.005):
            mode = MODE_ESCALATE
            internal_state["mode_entry_step"] = internal_state["current_step"]
        # Escalate to RECOVERY if risk remains high or FAIL zone persists
        elif (risk_ema >= brace_to_recovery_risk) or (
            tilt_zone == "FAIL" and prefall_fail_frac >= brace_to_recovery_pff
        ):
            mode = MODE_RECOVERY
            internal_state["mode_entry_step"] = internal_state["current_step"]
        # De-escalate to NORMAL when things are calm again
        elif (risk_ema <= brace_to_normal_risk) and (prefall_fail_frac <= brace_to_normal_pff):
            mode = MODE_NORMAL
            internal_state["mode_entry_step"] = internal_state["current_step"]
    
    # ESCALATE → RECOVERY or BRACE
    elif mode == MODE_ESCALATE:
        # If conditions worsen, go to RECOVERY
        if (risk_ema >= brace_to_recovery_risk) or (tilt_zone == "FAIL" and prefall_fail_frac >= brace_to_recovery_pff):
            mode = MODE_RECOVERY
            internal_state["mode_entry_step"] = internal_state["current_step"]
        # If conditions improve, go back to BRACE
        elif (risk_ema <= recovery_to_brace_risk) and (tilt_zone != "FAIL"):
            mode = MODE_BRACE
            internal_state["mode_entry_step"] = internal_state["current_step"]
    
    # RECOVERY → BRACE when risk has dropped enough (with minimum duration)
    elif mode == MODE_RECOVERY:
        steps_in_recovery = internal_state["current_step"] - internal_state.get("mode_entry_step", 0)
        min_recovery_duration = 40  # Minimum 40 steps in RECOVERY before switching back
        
        # Only allow transition if minimum duration met
        if steps_in_recovery >= min_recovery_duration:
            if risk_ema <= recovery_to_brace_risk:
                mode = MODE_BRACE
                internal_state["mode_entry_step"] = internal_state["current_step"]
    
    internal_state["edon_mode"] = mode
    
    # Track mode steps for debug (include ESCALATE)
    ms = internal_state.setdefault("mode_steps", {
        MODE_NORMAL: 0, MODE_BRACE: 0, MODE_ESCALATE: 0, MODE_RECOVERY: 0
    })
    ms[mode] = ms.get(mode, 0) + 1


def _get_mode_scaling(edon_mode: int) -> Tuple[float, float, float]:
    """
    Return (base_scale, prefall_mult, safe_mult) for the given EDON mode.
    These scalings modulate the final blended action.
    
    - base_scale: multiplies the baseline PD action
    - prefall_mult: multiplies the PREFALL torque term
    - safe_mult: multiplies the SAFE torque term
    
    Args:
        edon_mode: Current EDON mode (MODE_NORMAL, MODE_BRACE, or MODE_RECOVERY)
    
    Returns:
        Tuple of (base_scale, prefall_mult, safe_mult)
    """
    if edon_mode == MODE_NORMAL:
        # Normal: keep things mostly as-is
        base_scale   = 1.00
        prefall_mult = 1.00
        safe_mult    = 1.00
    
    elif edon_mode == MODE_BRACE:
        # Brace: minimal baseline damping, stronger stabilizing EDON torques
        base_scale   = 0.96   # INCREMENTAL: was 0.95 - minimal damping (4% reduction)
        prefall_mult = 1.30   # INCREMENTAL: was 1.25 - moderate 30% boost (testing sweet spot)
        safe_mult    = 1.35   # INCREMENTAL: was 1.30 - moderate 35% boost (testing sweet spot)
    
    elif edon_mode == MODE_ESCALATE:
        # Escalate: moderate damping, strongest corrections
        base_scale   = 0.70   # INCREMENTAL: was 0.65 - moderate damping (30% reduction)
        prefall_mult = 2.5    # INCREMENTAL: was 2.4 - moderate increase (testing sweet spot)
        safe_mult    = 2.8    # INCREMENTAL: was 2.7 - moderate increase (testing sweet spot)
    
    elif edon_mode == MODE_RECOVERY:
        # Recovery: moderate damping, strongest stabilizing torques
        base_scale   = 0.78   # INCREMENTAL: was 0.75 - moderate damping (22% reduction)
        prefall_mult = 2.2    # INCREMENTAL: was 2.0 - moderate increase (testing sweet spot)
        safe_mult    = 2.4    # INCREMENTAL: was 2.2 - moderate increase (testing sweet spot)
    
    else:
        # Fallback to NORMAL if something is off
        base_scale   = 1.00
        prefall_mult = 1.00
        safe_mult    = 1.00
    
    return base_scale, prefall_mult, safe_mult


def compute_multi_layer_correction(
    baseline_action: np.ndarray,
    tilt_info: Dict[str, Any],
    obs: Dict[str, Any]
) -> np.ndarray:
    """
    Compute multi-layer correction: tilt + velocity damping + gait smoothing.
    
    Args:
        baseline_action: Baseline action vector
        tilt_info: Tilt information dict
        obs: Full observation dict
    
    Returns:
        Raw correction vector (before scaling)
    """
    correction = np.zeros_like(baseline_action)
    
    roll = tilt_info["roll"]
    pitch = tilt_info["pitch"]
    roll_velocity = tilt_info["roll_velocity"]
    pitch_velocity = tilt_info["pitch_velocity"]
    
    # 1) Tilt correction on balance joints
    # PD: proportional (opposite to tilt) + derivative (opposite to tilt rate)
    corr_roll = -KP_ROLL * roll - KD_ROLL * roll_velocity
    corr_pitch = -KP_PITCH * pitch - KD_PITCH * pitch_velocity
    
    # Map to balance joints
    tilt_vec = np.zeros_like(baseline_action)
    if len(baseline_action) >= 4:
        tilt_vec[0] = corr_roll   # Roll correction
        tilt_vec[1] = corr_pitch  # Pitch correction
        tilt_vec[2] = corr_roll * 0.3   # COM x (smaller)
        tilt_vec[3] = corr_pitch * 0.3  # COM y (smaller)
    
    # 2) Velocity damping: damp high speeds in balance joints
    vel_damp_masked = np.zeros_like(baseline_action)
    if "joint_vel" in obs and len(obs["joint_vel"]) >= len(baseline_action):
        joint_vel = np.array(obs["joint_vel"][:len(baseline_action)])
        vel_damp = -VEL_DAMP_GAIN * joint_vel
        for j in BALANCE_JOINTS:
            if j < len(vel_damp_masked):
                vel_damp_masked[j] = vel_damp[j] if j < len(vel_damp) else 0.0
    
    # 3) Gait smoothing: very small smoothing against baseline (reduced to avoid destabilizing)
    gait_smooth = -0.01 * baseline_action  # FIX: was GAIT_SMOOTH_GAIN (0.05) - reduced to 0.01
    
    # Combine all layers
    raw_correction = tilt_vec + vel_damp_masked + gait_smooth
    
    # Only apply to balance joints (zero others)
    mask = np.zeros_like(baseline_action)
    for j in BALANCE_JOINTS:
        if j < len(mask):
            mask[j] = 1.0
    raw_correction *= mask
    
    return raw_correction


def check_tilt_direction(
    raw_correction: np.ndarray,
    tilt_info: Dict[str, Any],
    baseline_action: np.ndarray
) -> Tuple[np.ndarray, bool]:
    """
    Check if correction opposes tilt direction (not baseline).
    
    Args:
        raw_correction: Correction vector
        tilt_info: Tilt information dict
        baseline_action: Baseline action (for reference)
    
    Returns:
        Tuple of (corrected_vector, was_flipped)
    """
    roll = tilt_info["roll"]
    pitch = tilt_info["pitch"]
    
    # Define tilt direction vector in joint space
    # If pitch > 0 (falling forward), we want corrections that push back (negative)
    tilt_dir = np.zeros_like(baseline_action)
    if len(baseline_action) >= 4:
        tilt_dir[0] = roll   # Roll direction
        tilt_dir[1] = pitch  # Pitch direction
        tilt_dir[2] = roll * 0.3
        tilt_dir[3] = pitch * 0.3
    
    # Compute dot product
    dot = np.dot(raw_correction, tilt_dir)
    
    # If dot > 0, correction is in same direction as tilt → destabilizing
    was_flipped = False
    if dot > 0 and np.linalg.norm(tilt_dir) > 1e-6:
        raw_correction = -1.0 * raw_correction  # FIX: was -0.7 - flip fully (not reduced)
        was_flipped = True
    
    return raw_correction, was_flipped


def apply_edon_regulation_v3(
    baseline_action: np.ndarray,
    obs: Dict[str, Any],
    edon_output: Optional[Dict[str, Any]],
    edon_gain: float,
    internal_state: Dict[str, Any],
    debug_info: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    EDON Controller V3 - "Chaos Stabilizer"
    
    Multi-layer, risk-aware stabilizer that uses EDON's p_stress, p_chaos, and risk
    to intelligently shape corrections.
    
    Args:
        baseline_action: Original action from baseline controller
        obs: Full observation dictionary
        edon_output: Raw EDON engine output (contains p_stress, p_chaos, etc.)
        edon_gain: [0..1] global blend factor
        internal_state: Dict storing EMAs, hysteresis flags, etc. across steps
        debug_info: Dict to accumulate debug stats (may be None)
    
    Returns:
        Tuple of (final_action, updated_internal_state)
    """
    # Initialize internal state if needed
    if "risk_ema" not in internal_state:
        internal_state["risk_ema"] = 0.0
    if "prev_zone" not in internal_state:
        internal_state["prev_zone"] = "SAFE"
    # V3.5 state initialization
    if "edon_prev_risk" not in internal_state:
        internal_state["edon_prev_risk"] = 0.0
    if "prefall_ratio" not in internal_state:
        internal_state["prefall_ratio"] = EDON_V35_CFG["PREFALL_BASE"]
    if "safe_active" not in internal_state:
        internal_state["safe_active"] = False
    # V3.6 brace-mode state initialization (legacy - kept for compatibility)
    if "brace_mode" not in internal_state:
        internal_state["brace_mode"] = False
    if "brace_steps" not in internal_state:
        internal_state["brace_steps"] = 0
    if "total_steps" not in internal_state:
        internal_state["total_steps"] = 0
    # V3.7 multi-mode state initialization
    if "edon_mode" not in internal_state:
        internal_state["edon_mode"] = MODE_NORMAL
    if "mode_steps" not in internal_state:
        internal_state["mode_steps"] = {MODE_NORMAL: 0, MODE_BRACE: 0, MODE_ESCALATE: 0, MODE_RECOVERY: 0}
    if "mode_entry_step" not in internal_state:
        internal_state["mode_entry_step"] = 0
    if "current_step" not in internal_state:
        internal_state["current_step"] = 0
    if "zone_history" not in internal_state:
        internal_state["zone_history"] = []
    
    # Compute tilt information
    tilt_info = compute_tilt_info(obs)
    
    # Compute risk and update EMA
    risk_score = compute_risk_score(edon_output)
    alpha = 0.15
    risk_ema = (1 - alpha) * internal_state["risk_ema"] + alpha * risk_score
    internal_state["risk_ema"] = risk_ema
    
    # Initialize edon_prev_risk_ema if needed (for delta_ema computation)
    if "edon_prev_risk_ema" not in internal_state:
        internal_state["edon_prev_risk_ema"] = 0.0
    
    # Compute delta_ema for ESCALATE mode detection and FAIL anticipation
    delta_ema = risk_ema - internal_state["edon_prev_risk_ema"]
    
    # Compute internal zone (with virtual PREFALL and hysteresis)
    internal_zone = compute_internal_zone(
        tilt_info,
        risk_ema,
        internal_state["prev_zone"]
    )
    internal_state["prev_zone"] = internal_zone
    
    # Compute prefall_fail_frac (fraction of recent steps in PREFALL or FAIL)
    # Track last N zones for fraction calculation (simple window)
    if "zone_history" not in internal_state:
        internal_state["zone_history"] = []
    internal_state["zone_history"].append(internal_zone)
    # Keep last 20 steps
    if len(internal_state["zone_history"]) > 20:
        internal_state["zone_history"] = internal_state["zone_history"][-20:]
    prefall_fail_count = sum(1 for z in internal_state["zone_history"] if z in ["PREFALL", "FAIL"])
    prefall_fail_frac = prefall_fail_count / max(len(internal_state["zone_history"]), 1)
    
    # Update EDON mode based on risk_ema, tilt_zone, prefall_fail_frac, and delta_ema
    _update_edon_mode(internal_state, risk_ema, internal_zone, prefall_fail_frac, delta_ema)
    
    # Update edon_prev_risk_ema for next step (after using it for delta_ema)
    internal_state["edon_prev_risk_ema"] = risk_ema
    
    # FAIL anticipation: detect FAIL entry or risk spike
    fail_anticipation_active = False
    if internal_zone == "FAIL":
        fail_anticipation_active = True
    elif risk_ema > 0.050:  # Risk spike threshold
        fail_anticipation_active = True
    elif delta_ema > 0.008:  # Rapid risk increase
        fail_anticipation_active = True
    
    # Track mode steps
    current_mode = internal_state["edon_mode"]
    internal_state["mode_steps"][current_mode] = internal_state["mode_steps"].get(current_mode, 0) + 1
    
    # Get EDON state for modulation
    edon_state_enum = map_edon_output_to_state(edon_output) if edon_output else None
    if edon_state_enum is not None:
        internal_state["last_state"] = edon_state_enum
    
    # Build edon_state dict for adaptive functions
    # Extract instability, disturbance, and phase from EDON output and internal state
    p_stress = edon_output.get("p_stress", 0.0) if edon_output else 0.0
    p_chaos = edon_output.get("p_chaos", 0.0) if edon_output else 0.0
    
    # Instability score: combination of chaos and stress
    instability_score = 0.6 * p_chaos + 0.4 * p_stress
    
    # Disturbance level: based on risk EMA
    disturbance_level = risk_ema
    
    # Phase: based on internal zone
    if internal_zone == "FAIL":
        phase = "recovery"
    elif internal_zone == "PREFALL":
        phase = "pre_fall"
    else:
        phase = "normal"
    
    # Fall risk: use risk_ema as fall risk predictor
    fall_risk = risk_ema
    
    # Catastrophic risk: very high risk or in FAIL zone
    catastrophic_risk = 1.0 if internal_zone == "FAIL" else (risk_ema * 1.2)  # Scale up risk_ema
    catastrophic_risk = clamp(catastrophic_risk, 0.0, 1.0)
    
    # === PREDICTED INSTABILITY (V5.2) ===
    # Super-conservative: use delta_ema as proxy, scale gently
    # delta_ema * 2.0, capped at 0.5 (not too spicy)
    predicted_instability = max(0.0, min(delta_ema * 2.0, 0.6))
    
    # Build edon_state dict for adaptive functions
    edon_state_dict = {
        "instability_score": instability_score,
        "disturbance_level": disturbance_level,
        "phase": phase,
        "fall_risk": fall_risk,
        "catastrophic_risk": catastrophic_risk,
        "predicted_instability": predicted_instability  # V5.2: mild predicted boost
    }
    
    # Compute base PD correction (EDON correction vector)
    # This is the correction that EDON wants to apply
    edon_correction = compute_multi_layer_correction(baseline_action, tilt_info, obs)
    
    # Check direction vs tilt
    edon_correction, was_flipped = check_tilt_direction(edon_correction, tilt_info, baseline_action)
    
    # Compute PREFALL direction (same as edon_correction for now, but could be different)
    prefall_direction = edon_correction.copy()
    
    # Compute SAFE posture torque (stabilizing correction)
    safe_posture_torque = np.zeros_like(baseline_action)
    roll = tilt_info["roll"]
    pitch = tilt_info["pitch"]
    safe_posture_torque[0] = -0.15 * roll   # Stronger safe correction when activated
    safe_posture_torque[1] = -0.15 * pitch
    # Only apply to balance joints
    mask = np.zeros_like(baseline_action)
    for j in BALANCE_JOINTS:
        if j < len(mask):
            mask[j] = 1.0
    safe_posture_torque *= mask
    
    # Apply state-aware adaptive EDON gain
    torque_cmd = apply_edon_gain(baseline_action, edon_correction, edon_state_dict)
    
    # Apply PREFALL reflex (dynamic, risk-based, with decay)
    torque_cmd = apply_prefall_reflex(torque_cmd, edon_state_dict, prefall_direction, internal_state)
    
    # Apply SAFE override (only when catastrophic risk is high)
    torque_cmd = apply_safe_override(torque_cmd, edon_state_dict, safe_posture_torque)
    
    # === STABILITY-WEIGHTED LOW-PASS FILTER (V5.1) ===
    # This kills noise, over-corrections, and makes corrections coherent
    # When stable → more smoothing (prevents twitch)
    # When unstable → less smoothing (still responsive)
    if "torque_cmd_prev" not in internal_state:
        internal_state["torque_cmd_prev"] = baseline_action.copy()
    
    # Alpha: V5.1 FINAL - Selected from incremental testing
    # Test 1 (0.75 - 0.15) was best: +0.2% ± 0.2% (only positive result)
    # Stable (instability=0): alpha=0.75 (75% smoothing)
    # Unstable (instability=1): alpha=0.60 (60% smoothing)
    alpha = clamp(0.75 - 0.15 * instability_score, 0.4, 0.9)
    torque_cmd = alpha * internal_state["torque_cmd_prev"] + (1.0 - alpha) * torque_cmd
    
    # Store for next step
    internal_state["torque_cmd_prev"] = torque_cmd.copy()
    
    final_action = torque_cmd
    
    # Track actual ratio for debug (approximate)
    baseline_norm = np.linalg.norm(baseline_action) + 1e-8
    correction_norm = np.linalg.norm(edon_correction) + 1e-8
    final_prefall_ratio = (fall_risk * 0.35 + 0.10) * (correction_norm / baseline_norm) if baseline_norm > 0 else 0.0
    
    # Track final ratio for debug (PREFALL ratio)
    final_ratio = final_prefall_ratio
    
    # Update debug info
    if debug_info is not None:
        # Zone counts
        zone_key = f"zone_{internal_zone.lower()}_count"
        debug_info[zone_key] = debug_info.get(zone_key, 0) + 1
        
        # Tilt zone counts (geometric)
        geom_zone_key = f"tilt_zone_{tilt_info['zone'].lower()}_count"
        debug_info[geom_zone_key] = debug_info.get(geom_zone_key, 0) + 1
        
        # Correction ratios per zone (track final ratio after scaling)
        ratio_key = f"ratio_{internal_zone.lower()}_sum"
        debug_info[ratio_key] = debug_info.get(ratio_key, 0.0) + final_ratio
        
        # Step counts per zone
        steps_key = f"corr_steps_{internal_zone.lower()}"
        debug_info[steps_key] = debug_info.get(steps_key, 0) + 1
        
        # Flipped corrections
        if was_flipped:
            debug_info["num_flipped_corrections"] = debug_info.get("num_flipped_corrections", 0) + 1
        
        # Risk EMA stats
        debug_info["risk_ema_sum"] = debug_info.get("risk_ema_sum", 0.0) + risk_ema
        debug_info["risk_ema_max"] = max(debug_info.get("risk_ema_max", 0.0), risk_ema)
        debug_info["risk_ema_count"] = debug_info.get("risk_ema_count", 0) + 1
        
        # V3.5 stats - track min/max for real runtime values
        current_prefall_ratio = internal_state.get("prefall_ratio", 0.0)
        debug_info["prefall_ratio_sum"] = debug_info.get("prefall_ratio_sum", 0.0) + current_prefall_ratio
        debug_info["prefall_ratio_count"] = debug_info.get("prefall_ratio_count", 0) + 1
        # Track min/max - initialize on first step
        if debug_info.get("prefall_ratio_min") is None:
            debug_info["prefall_ratio_min"] = current_prefall_ratio
            debug_info["prefall_ratio_max"] = current_prefall_ratio
        else:
            debug_info["prefall_ratio_min"] = min(debug_info["prefall_ratio_min"], current_prefall_ratio)
            debug_info["prefall_ratio_max"] = max(debug_info["prefall_ratio_max"], current_prefall_ratio)
        if internal_state.get("safe_active", False):
            debug_info["safe_active_steps"] = debug_info.get("safe_active_steps", 0) + 1
        # V3.6 brace-mode stats (legacy - kept for compatibility)
        if internal_state.get("brace_mode", False):
            debug_info["brace_mode_steps"] = debug_info.get("brace_mode_steps", 0) + 1
        # V3.7 multi-mode stats
        current_mode = internal_state.get("edon_mode", MODE_NORMAL)
        mode_steps = internal_state.get("mode_steps", {})
        if "mode_steps" not in debug_info:
            debug_info["mode_steps"] = {MODE_NORMAL: 0, MODE_BRACE: 0, MODE_RECOVERY: 0}
        debug_info["mode_steps"][current_mode] = debug_info["mode_steps"].get(current_mode, 0) + 1
    
    return final_action, internal_state

