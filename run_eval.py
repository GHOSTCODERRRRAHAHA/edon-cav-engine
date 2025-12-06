#!/usr/bin/env python3
"""
EDON Humanoid Evaluation Runner

Runs A/B tests comparing baseline vs EDON-enabled control policies.
Tracks metrics: interventions, freezes, stability, episode length, success rate.

V4.1: Uses adaptive gain system with BASE_GAIN=0.50, dynamic PREFALL, catastrophic-only SAFE.
V4 Adaptive: Stateful, context-aware adaptive controller with phase-based logic.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

# Evaluation imports
from evaluation.humanoid_runner import HumanoidRunner
from evaluation.metrics import EpisodeMetrics, RunMetrics, aggregate_run_metrics
from evaluation.config import config
from evaluation.mock_env import MockHumanoidEnv
from evaluation.randomization import EnvironmentRandomizer
from evaluation.stress_profiles import get_stress_profile, STRESS_PROFILES
from evaluation.edon_state import EdonState, map_edon_output_to_state, compute_risk_score
from training.edon_score import compute_episode_score

# EDON SDK
try:
    from edon import EdonClient
except ImportError:
    try:
        from sdk.python.edon.client import EdonClient
    except ImportError:
        print("WARNING: EDON SDK not found. EDON mode will not work.")
        EdonClient = None

# ============================================================================
# EDON Controller Configuration (loadable from JSON)
# ============================================================================

# Default EDON controller constants
_DEFAULT_EDON_CONFIG = {
    "GAIN_STABLE": 0.9,
    "GAIN_WARNING": 1.0,
    "GAIN_RECOVERY": 1.1,
    "CLAMP_RATIO_STABLE": 1.20,
    "CLAMP_RATIO_WARNING": 1.20,
    "CLAMP_RATIO_RECOVERY": 1.50,
    "W_PREFALL_STABLE": 0.3,
    "W_SAFE_STABLE": 0.7,
    "W_PREFALL_WARNING": 0.5,
    "W_SAFE_WARNING": 0.5,
    "W_PREFALL_RECOVERY": 0.5,
    "W_SAFE_RECOVERY": 0.5,
}

# Load config from environment variable if set
_EDON_CONFIG = _DEFAULT_EDON_CONFIG.copy()
cfg_path = os.getenv("EDON_CONFIG_PATH")
if cfg_path and os.path.exists(cfg_path):
    try:
        with open(cfg_path, "r") as f:
            user_cfg = json.load(f)
        # Override defaults with user config (only keys that exist in defaults)
        for key in _DEFAULT_EDON_CONFIG.keys():
            if key in user_cfg:
                _EDON_CONFIG[key] = user_cfg[key]
    except Exception as e:
        print(f"[WARNING] Failed to load EDON config from {cfg_path}: {e}")
        print(f"[WARNING] Using default config values.")

# ============================================================================
# Controller Functions
# ============================================================================

def baseline_controller(obs: dict, edon_state: Optional[dict] = None) -> np.ndarray:
    """
    Baseline control policy (EDON off).
    
    Simple PD-like controller that tries to maintain balance.
    """
    # Extract state
    roll = obs.get("roll", 0.0)
    pitch = obs.get("pitch", 0.0)
    com_x = obs.get("com_x", 0.0)
    com_y = obs.get("com_y", 0.0)
    
    # Simple balance controller: apply torque proportional to tilt
    action_size = 10  # Adjust to match your action space
    action = np.zeros(action_size)
    
    # Apply corrective torques based on orientation
    action[0] = -roll * 0.5  # Correct roll
    action[1] = -pitch * 0.5  # Correct pitch
    action[2] = -com_x * 0.3  # Correct COM x
    action[3] = -com_y * 0.3  # Correct COM y
    
    # Add some exploration noise (makes baseline less stable)
    action += np.random.normal(0, 0.1, size=action_size)
    
    # Clip to reasonable range
    action = np.clip(action, -1.0, 1.0)
    
    return action


# ============================================================================
# EDON Controller Hyperparameters (V4.1 - Optimized for 10% improvement)
# Based on +7.0% result: BASE_GAIN=0.5, PREFALL=0.15-0.60
# Tuned: Kp increased to 0.14, PREFALL_MAX to 0.65 for stronger corrections
# ============================================================================
EDON_BASE_KP_ROLL = 0.15  # Increased from 0.08 (1.875x stronger, balanced)
EDON_BASE_KP_PITCH = 0.15  # Increased from 0.08 (1.875x stronger)
EDON_BASE_KD_ROLL = 0.04  # Increased from 0.02 (2x stronger)
EDON_BASE_KD_PITCH = 0.04  # Increased from 0.02 (2x stronger)
# Phase-dependent clamp ratios (replaces static EDON_MAX_CORRECTION_RATIO)
# These are defined in EDONController and used dynamically

FORWARD_VELOCITY_INDICES = [4, 5, 6, 7, 8, 9]  # Movement actions


# ============================================================================
# EDON v5 Architecture - Modular Three-Layer Design
# ============================================================================

@dataclass
class EdonFeatures:
    """
    Extracted scalar features for EDON decision-making.
    
    Contains all computed features used by the controller for instability
    scoring, phase transitions, and policy decisions.
    """
    tilt_mag: float = 0.0
    vel_norm: float = 0.0
    p_chaos: float = 0.0
    p_stress: float = 0.0
    risk_ema: float = 0.0
    tilt_zone: str = "safe"  # "safe", "prefall", "fail"
    internal_zone: str = "safe"  # Internal zone classification
    roll: float = 0.0
    pitch: float = 0.0
    roll_velocity: float = 0.0
    pitch_velocity: float = 0.0
    max_tilt: float = 0.0
    risk_score: float = 0.0


@dataclass
class EdonCoreState:
    """
    Core state for EDON v5 controller.
    
    Tracks phase, instability, gain, and smoothed signals for the
    state machine and policy decisions.
    """
    phase: str = "stable"  # "stable", "warning", "recovery"
    instability_score: float = 0.0
    adaptive_gain: float = 0.0
    fail_risk: float = 0.0  # Predicted failure risk (v8)
    step_count: int = 0
    episode_id: int = -1
    
    # EMAs
    smoothed_tilt: float = 0.0
    smoothed_vel_norm: float = 0.0
    smoothed_instability: float = 0.0
    last_gain: float = 0.0
    
    # Phase tracking for logging
    phase_counts: Dict[str, int] = field(default_factory=lambda: {"stable": 0, "warning": 0, "recovery": 0})
    instability_history: list = field(default_factory=list)
    gain_history: list = field(default_factory=list)
    
    # Diagnostic tracking
    prefall_magnitudes: list = field(default_factory=list)
    safe_magnitudes: list = field(default_factory=list)
    delta_magnitudes: list = field(default_factory=list)
    prefall_active_count: int = 0
    safe_active_count: int = 0


class EdonPolicyBase:
    """
    Base class for EDON policy implementations.
    
    Policies compute the action delta (EDON contribution) given features
    and core state. The delta is added to baseline action to get final action.
    """
    def compute_delta(
        self,
        features: EdonFeatures,
        core_state: EdonCoreState,
        baseline_action: np.ndarray
    ) -> np.ndarray:
        """
        Compute EDON action delta (contribution to add to baseline).
        
        Args:
            features: Extracted features from current observation
            core_state: Current core state (phase, gain, etc.)
            baseline_action: Baseline controller action
            
        Returns:
            Action delta (same shape as baseline_action)
        """
        raise NotImplementedError("Subclasses must implement compute_delta")


# ============================================================================
# EDON v4 Adaptive Layer - Stateful, Context-Aware Controller
# (Kept for backward compatibility, will be wrapped by v5)
# ============================================================================

@dataclass
class EDONAdaptiveState:
    """
    Internal adaptive state for EDON controller.
    
    Tracks instability signals, phase transitions, and smoothed metrics
    to enable context-aware adaptation.
    """
    last_gain: float = 0.0
    instability_score: float = 0.0
    phase: str = "stable"  # "stable", "warning", "recovery"
    smoothed_tilt: float = 0.0
    smoothed_vel_norm: float = 0.0
    smoothed_interventions: float = 0.0
    episode_id: int = -1
    step_count: int = 0
    
    # Phase tracking for logging
    phase_counts: Dict[str, int] = field(default_factory=lambda: {"stable": 0, "warning": 0, "recovery": 0})
    instability_history: list = field(default_factory=list)
    gain_history: list = field(default_factory=list)
    
    # Diagnostic tracking for torque activity
    prefall_magnitudes: list = field(default_factory=list)  # Magnitudes when prefall is active
    safe_magnitudes: list = field(default_factory=list)  # Magnitudes when safe is active
    delta_magnitudes: list = field(default_factory=list)  # Action delta magnitudes
    prefall_active_count: int = 0  # Steps where prefall torque is active
    safe_active_count: int = 0  # Steps where safe torque is active


class EDONController:
    """
    EDON v4 Adaptive Controller
    
    ============================================================================
    ADAPTIVE STRATEGY (Plain Language)
    ============================================================================
    
    This controller transforms EDON from a static gain layer into a stateful,
    context-aware adaptive intelligence layer. Instead of applying the same
    correction formula every step, it:
    
    1. MONITORS INSTABILITY: Continuously computes an instability score from:
       - Torso tilt magnitude (how far off-balance)
       - Joint velocities (how fast it's moving)
       - EDON risk signals (p_chaos, p_stress from EDON API)
       - Tilt zone classification (safe/prefall/fail)
    
    2. TRACKS PHASES: Maintains three operational phases:
       - STABLE: Normal operation, low instability
       - WARNING: Elevated instability, needs attention
       - RECOVERY: High instability, aggressive correction needed
    
    3. ADAPTS GAIN: Dynamically adjusts correction strength:
       - Stable phase: 50% of base gain (gentle, energy-efficient)
       - Warning phase: 100% of base gain (normal response)
       - Recovery phase: 110% of base gain, capped at 1.1 (prevents overpowering baseline)
       - Uses exponential smoothing to prevent sudden jumps
    
    4. BLENDS TORQUES: Mixes two correction types based on phase:
       - Stable: 30% prefall_torque, 70% safe_torque (preventive, gentle)
       - Warning: 60% prefall_torque, 40% safe_torque (balanced response)
       - Recovery: 55% prefall_torque, 45% safe_torque (balanced, prevents overcorrection)
    
    5. PREVENTS OSCILLATION: Uses hysteresis (different thresholds for
       entering vs leaving phases) and exponential moving averages to smooth
       noisy signals, preventing the controller from flipping phases every frame.
    
    The result is a controller that is gentle when stable, responsive when
    warning, and aggressive when recovering, all while maintaining smooth
    transitions and avoiding instability.
    """
    
    def __init__(self, base_edon_gain: float = 0.75):
        """
        Initialize EDON controller.
        
        Args:
            base_edon_gain: Base gain from CLI (scales entire EDON contribution)
        """
        self.base_edon_gain = base_edon_gain
        self.adaptive_state = EDONAdaptiveState()
        
        # Phase transition thresholds (with hysteresis)
        # Lowered recovery threshold to allow actual recovery mode engagement
        self.T_WARNING_ON = 0.42   # stable -> warning when instability > this (10% attempt: lowered to 0.42)
        self.T_WARNING_OFF = 0.3  # warning -> stable when instability < this
        self.T_RECOVERY_ON = 0.58  # warning -> recovery when instability > this (lowered from 0.85)
        self.T_RECOVERY_OFF = 0.30  # recovery -> warning when instability < this (raised from 0.2)
        
        # Smoothing parameters
        self.ALPHA_TILT = 0.2      # EMA for tilt magnitude
        self.ALPHA_VEL = 0.2       # EMA for velocity norm
        self.ALPHA_GAIN = 0.65     # EMA for adaptive gain (increased from 0.2 to let base gain matter)
        self.ALPHA_INSTABILITY = 0.15  # EMA for instability score
        
        # Phase-based gain multipliers
        # Recovery multiplier kept near 1.1 to let base gain drive authority
        # Load from config (set via EDON_CONFIG_PATH environment variable)
        self.GAIN_STABLE = _EDON_CONFIG["GAIN_STABLE"]
        self.GAIN_WARNING = _EDON_CONFIG["GAIN_WARNING"]
        self.GAIN_RECOVERY = _EDON_CONFIG["GAIN_RECOVERY"]
        self.GAIN_RECOVERY_MAX = 1.1  # Absolute maximum recovery gain (fixed)
        
        # Phase-based clamp ratios (replaces static EDON_MAX_CORRECTION_RATIO)
        # Allows EDON more authority when recovering, safety when stable
        # Load from config
        self.CLAMP_RATIO_STABLE = _EDON_CONFIG["CLAMP_RATIO_STABLE"]
        self.CLAMP_RATIO_WARNING = _EDON_CONFIG["CLAMP_RATIO_WARNING"]
        self.CLAMP_RATIO_RECOVERY = _EDON_CONFIG["CLAMP_RATIO_RECOVERY"]
        
        # Phase-based torque blending weights
        # More conservative blending to prevent overcorrection
        # Load from config
        self.W_PREFALL_STABLE = _EDON_CONFIG["W_PREFALL_STABLE"]
        self.W_SAFE_STABLE = _EDON_CONFIG["W_SAFE_STABLE"]
        self.W_PREFALL_WARNING = _EDON_CONFIG["W_PREFALL_WARNING"]
        self.W_SAFE_WARNING = _EDON_CONFIG["W_SAFE_WARNING"]
        self.W_PREFALL_RECOVERY = _EDON_CONFIG["W_PREFALL_RECOVERY"]
        self.W_SAFE_RECOVERY = _EDON_CONFIG["W_SAFE_RECOVERY"]
        
        # Logging
        self.logging_enabled = True
        self._logged_config = False
    
    def compute_instability_score(
        self,
        obs: Dict[str, Any],
        edon_state_raw: Optional[Dict[str, Any]],
        tilt_zone: str,
        risk_ema: float
    ) -> float:
        """
        Compute instability score from current state.
        
        Combines:
        - Torso tilt magnitude
        - Joint velocity norm
        - EDON risk signals (p_chaos, p_stress)
        - Tilt zone classification
        
        Returns:
            Instability score in [0, 1] range (higher = more unstable)
        """
        # Extract tilt
        roll = obs.get("roll", 0.0)
        pitch = obs.get("pitch", 0.0)
        tilt_magnitude = np.sqrt(roll**2 + pitch**2)
        
        # Extract velocities
        roll_velocity = obs.get("roll_velocity", 0.0)
        pitch_velocity = obs.get("pitch_velocity", 0.0)
        vel_norm = np.sqrt(roll_velocity**2 + pitch_velocity**2)
        
        # Normalize tilt (0-1 scale, assuming max reasonable tilt ~0.4 rad)
        tilt_normalized = min(tilt_magnitude / 0.4, 1.0)
        
        # Normalize velocity (0-1 scale, assuming max reasonable velocity ~5.0 rad/s)
        vel_normalized = min(vel_norm / 5.0, 1.0)
        
        # EDON risk signals
        p_chaos = 0.0
        p_stress = 0.0
        if edon_state_raw is not None:
            p_chaos = max(0.0, min(1.0, edon_state_raw.get("p_chaos", 0.0)))
            p_stress = max(0.0, min(1.0, edon_state_raw.get("p_stress", 0.0)))
        
        # Combine signals with weights
        # Tilt is most important (direct instability indicator)
        # Velocity indicates dynamic instability
        # EDON signals provide predictive risk
        # Tilt zone provides context
        instability = (
            0.35 * tilt_normalized +
            0.25 * vel_normalized +
            0.20 * p_chaos +
            0.15 * p_stress +
            0.05 * risk_ema
        )
        
        # Boost if in prefall/fail zone (reduced boost to prevent overreaction)
        if tilt_zone in ("prefall", "fail"):
            instability = min(1.0, instability * 1.15)  # Reduced from 1.3 to 1.15
        
        return max(0.0, min(1.0, instability))
    
    def update_phase(self, instability_score: float) -> str:
        """
        Update phase based on instability score with hysteresis.
        
        Phase transitions:
        - stable -> warning: when instability > T_WARNING_ON
        - warning -> recovery: when instability > T_RECOVERY_ON
        - recovery -> warning: when instability < T_RECOVERY_OFF
        - warning -> stable: when instability < T_WARNING_OFF
        
        Args:
            instability_score: Current instability score
            
        Returns:
            New phase string
        """
        current_phase = self.adaptive_state.phase
        
        if current_phase == "stable":
            if instability_score > self.T_WARNING_ON:
                new_phase = "warning"
            else:
                new_phase = "stable"
        elif current_phase == "warning":
            if instability_score > self.T_RECOVERY_ON:
                new_phase = "recovery"
            elif instability_score < self.T_WARNING_OFF:
                new_phase = "stable"
            else:
                new_phase = "warning"
        else:  # recovery
            if instability_score < self.T_RECOVERY_OFF:
                new_phase = "warning"
            else:
                new_phase = "recovery"
        
        # Update phase counts
        if new_phase != current_phase:
            self.adaptive_state.phase = new_phase
        self.adaptive_state.phase_counts[new_phase] = self.adaptive_state.phase_counts.get(new_phase, 0) + 1
        
        return new_phase
    
    def compute_adaptive_gain(self) -> float:
        """
        Compute adaptive gain based on current phase.
        
        Uses phase-dependent multipliers and smooths with EMA.
        
        Returns:
            Adaptive gain value
        """
        phase = self.adaptive_state.phase
        
        # Phase-based gain multiplier
        if phase == "stable":
            phase_gain = self.GAIN_STABLE
        elif phase == "warning":
            phase_gain = self.GAIN_WARNING
        else:  # recovery
            phase_gain = self.GAIN_RECOVERY
        
        # Compute raw adaptive gain
        adaptive_gain = self.base_edon_gain * phase_gain
        
        # Cap recovery gain to prevent EDON from overpowering baseline
        if phase == "recovery":
            adaptive_gain = min(adaptive_gain, self.GAIN_RECOVERY_MAX)
        
        # Smooth with previous gain to avoid sharp jumps
        last_gain = self.adaptive_state.last_gain
        if last_gain > 0.0:
            adaptive_gain = self.ALPHA_GAIN * adaptive_gain + (1 - self.ALPHA_GAIN) * last_gain
        
        # Store for next iteration
        self.adaptive_state.last_gain = adaptive_gain
        
        return adaptive_gain
    
    def get_torque_weights(self) -> Tuple[float, float]:
        """
        Get prefall and safe torque blending weights based on current phase.
        
        Returns:
            (w_prefall, w_safe) tuple
        """
        phase = self.adaptive_state.phase
        
        if phase == "stable":
            return (self.W_PREFALL_STABLE, self.W_SAFE_STABLE)
        elif phase == "warning":
            return (self.W_PREFALL_WARNING, self.W_SAFE_WARNING)
        else:  # recovery
            return (self.W_PREFALL_RECOVERY, self.W_SAFE_RECOVERY)
    
    def get_clamp_ratio(self) -> float:
        """
        Get phase-dependent clamp ratio for action limiting.
        
        Returns:
            Maximum ratio of EDON action magnitude to baseline action magnitude
        """
        phase = self.adaptive_state.phase
        
        if phase == "stable":
            return self.CLAMP_RATIO_STABLE
        elif phase == "warning":
            return self.CLAMP_RATIO_WARNING
        else:  # recovery
            return self.CLAMP_RATIO_RECOVERY
    
    def update_smoothed_signals(
        self,
        tilt_magnitude: float,
        vel_norm: float,
        instability_score: float
    ):
        """Update exponential moving averages for smoothed signals."""
        # Smooth tilt
        self.adaptive_state.smoothed_tilt = (
            self.ALPHA_TILT * tilt_magnitude +
            (1 - self.ALPHA_TILT) * self.adaptive_state.smoothed_tilt
        )
        
        # Smooth velocity
        self.adaptive_state.smoothed_vel_norm = (
            self.ALPHA_VEL * vel_norm +
            (1 - self.ALPHA_VEL) * self.adaptive_state.smoothed_vel_norm
        )
        
        # Smooth instability
        self.adaptive_state.instability_score = (
            self.ALPHA_INSTABILITY * instability_score +
            (1 - self.ALPHA_INSTABILITY) * self.adaptive_state.instability_score
        )
    
    def log_episode_summary(self, episode_id: int):
        """Log adaptive behavior summary for an episode."""
        if not self.logging_enabled:
            return
        
        # Log config on first episode
        if not self._logged_config and episode_id == 0:
            print(f"[EDON-ADAPTIVE] Configuration:")
            print(f"  base_gain={self.base_edon_gain:.3f}")
            print(f"  thresholds: warning_on={self.T_WARNING_ON:.2f}, warning_off={self.T_WARNING_OFF:.2f}")
            print(f"  thresholds: recovery_on={self.T_RECOVERY_ON:.2f}, recovery_off={self.T_RECOVERY_OFF:.2f}")
            self._logged_config = True
        
        # Compute episode averages
        if len(self.adaptive_state.instability_history) > 0:
            avg_instability = np.mean(self.adaptive_state.instability_history)
        else:
            avg_instability = 0.0
        
        if len(self.adaptive_state.gain_history) > 0:
            avg_gain = np.mean(self.adaptive_state.gain_history)
        else:
            avg_gain = self.base_edon_gain
        
        # Compute torque diagnostics
        total_steps = self.adaptive_state.step_count
        if len(self.adaptive_state.prefall_magnitudes) > 0:
            avg_prefall = np.mean(self.adaptive_state.prefall_magnitudes)
        else:
            avg_prefall = 0.0
        
        if len(self.adaptive_state.safe_magnitudes) > 0:
            avg_safe = np.mean(self.adaptive_state.safe_magnitudes)
        else:
            avg_safe = 0.0
        
        if len(self.adaptive_state.delta_magnitudes) > 0:
            avg_delta = np.mean(self.adaptive_state.delta_magnitudes)
        else:
            avg_delta = 0.0
        
        # Compute activity percentages
        prefall_active_pct = (self.adaptive_state.prefall_active_count / total_steps * 100.0) if total_steps > 0 else 0.0
        safe_active_pct = (self.adaptive_state.safe_active_count / total_steps * 100.0) if total_steps > 0 else 0.0
        
        # Log summary with diagnostics
        phase_counts_str = ", ".join([f"{k}={v}" for k, v in self.adaptive_state.phase_counts.items()])
        print(
            f"[EDON-ADAPTIVE] episode={episode_id} "
            f"phase_counts=({phase_counts_str}) "
            f"avg_instability={avg_instability:.3f} "
            f"avg_gain={avg_gain:.3f} "
            f"avg_prefall={avg_prefall:.4f} "
            f"avg_safe={avg_safe:.4f} "
            f"avg_delta={avg_delta:.4f} "
            f"prefall_active={prefall_active_pct:.1f}% "
            f"safe_active={safe_active_pct:.1f}%"
        )
        
        # Reset episode tracking
        self.adaptive_state.phase_counts = {"stable": 0, "warning": 0, "recovery": 0}
        self.adaptive_state.instability_history = []
        self.adaptive_state.gain_history = []
        self.adaptive_state.prefall_magnitudes = []
        self.adaptive_state.safe_magnitudes = []
        self.adaptive_state.delta_magnitudes = []
        self.adaptive_state.prefall_active_count = 0
        self.adaptive_state.safe_active_count = 0
    
    def reset_episode(self, episode_id: int):
        """Reset state for new episode."""
        self.adaptive_state.episode_id = episode_id
        self.adaptive_state.step_count = 0
        # Keep smoothed signals (they provide continuity)
        # Reset phase to stable
        self.adaptive_state.phase = "stable"
        self.adaptive_state.phase_counts = {"stable": 0, "warning": 0, "recovery": 0}
        # Reset diagnostic tracking
        self.adaptive_state.prefall_magnitudes = []
        self.adaptive_state.safe_magnitudes = []
        self.adaptive_state.delta_magnitudes = []
        self.adaptive_state.prefall_active_count = 0
        self.adaptive_state.safe_active_count = 0


# ============================================================================
# EDON v5 Policy Implementations
# ============================================================================

# ============================================================================
# EDON v6 Training Logger
# ============================================================================

class EdonTrainLogger:
    """
    Logger for EDON training data (JSONL format).
    
    Writes per-step data for offline learning.
    """
    
    def __init__(self, path: str, run_metadata: Dict[str, Any]):
        """
        Initialize training logger.
        
        Args:
            path: Path to JSONL file
            run_metadata: Metadata about the run (profile, episodes, seed, etc.)
        """
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")
        
        # Write metadata header
        meta_record = {"type": "meta", **run_metadata}
        self.f.write(json.dumps(meta_record) + "\n")
        self.f.flush()
    
    def log_step(self, record: Dict[str, Any]):
        """
        Log a single step record.
        
        Args:
            record: Step data dict (will be wrapped with type="step")
        """
        record = {"type": "step", **record}
        self.f.write(json.dumps(record) + "\n")
        self.f.flush()  # Flush for safety
    
    def log_episode_summary(self, record: Dict[str, Any]):
        """
        Log episode summary.
        
        Args:
            record: Episode summary dict (will be wrapped with type="episode_summary")
        """
        record = {"type": "episode_summary", **record}
        self.f.write(json.dumps(record) + "\n")
        self.f.flush()
    
    def close(self):
        """Close the log file."""
        if self.f is not None:
            self.f.close()
            self.f = None


# ============================================================================
# EDON v5 Policy Implementations
# ============================================================================

class HeuristicEdonPolicy(EdonPolicyBase):
    """
    Heuristic EDON policy wrapping v4.2 logic.
    
    This policy implements the existing heuristic-based torque blending
    and phase-dependent gain modulation. Behavior is identical to v4.2.
    """
    
    def __init__(
        self,
        gain_stable: float = 0.9,
        gain_warning: float = 1.0,
        gain_recovery: float = 1.1,
        clamp_ratio_stable: float = 1.20,
        clamp_ratio_warning: float = 1.20,
        clamp_ratio_recovery: float = 1.50,
        w_prefall_stable: float = 0.3,
        w_safe_stable: float = 0.7,
        w_prefall_warning: float = 0.5,
        w_safe_warning: float = 0.5,
        w_prefall_recovery: float = 0.5,
        w_safe_recovery: float = 0.5,
    ):
        """Initialize heuristic policy with configurable parameters."""
        self.gain_stable = gain_stable
        self.gain_warning = gain_warning
        self.gain_recovery = gain_recovery
        self.clamp_ratio_stable = clamp_ratio_stable
        self.clamp_ratio_warning = clamp_ratio_warning
        self.clamp_ratio_recovery = clamp_ratio_recovery
        self.w_prefall_stable = w_prefall_stable
        self.w_safe_stable = w_safe_stable
        self.w_prefall_warning = w_prefall_warning
        self.w_safe_warning = w_safe_warning
        self.w_prefall_recovery = w_prefall_recovery
        self.w_safe_recovery = w_safe_recovery
    
    def get_torque_weights(self, phase: str) -> Tuple[float, float]:
        """Get phase-based torque blending weights."""
        if phase == "stable":
            return (self.w_prefall_stable, self.w_safe_stable)
        elif phase == "warning":
            return (self.w_prefall_warning, self.w_safe_warning)
        else:  # recovery
            return (self.w_prefall_recovery, self.w_safe_recovery)
    
    def get_clamp_ratio(self, phase: str) -> float:
        """Get phase-based clamp ratio."""
        if phase == "stable":
            return self.clamp_ratio_stable
        elif phase == "warning":
            return self.clamp_ratio_warning
        else:  # recovery
            return self.clamp_ratio_recovery
    
    def compute_delta(
        self,
        features: EdonFeatures,
        core_state: EdonCoreState,
        baseline_action: np.ndarray
    ) -> np.ndarray:
        """
        Compute EDON delta using v4.2 heuristic logic.
        
        This is a direct port of the existing apply_edon_regulation logic,
        but returns only the delta (not final action).
        """
        # Extract values from features
        roll = features.roll
        pitch = features.pitch
        roll_velocity = features.roll_velocity
        pitch_velocity = features.pitch_velocity
        internal_zone = features.internal_zone
        risk_ema = features.risk_ema
        
        # ============================
        # COMPUTE BASE PD CORRECTIONS
        # ============================
        
        Kp_roll = EDON_BASE_KP_ROLL
        Kp_pitch = EDON_BASE_KP_PITCH
        Kd_roll = EDON_BASE_KD_ROLL
        Kd_pitch = EDON_BASE_KD_PITCH
        
        # Compute corrections with proper sign (oppose tilt)
        corr_roll = -Kp_roll * roll - Kd_roll * roll_velocity
        corr_pitch = -Kp_pitch * pitch - Kd_pitch * pitch_velocity
        
        # Ensure corrections oppose tilt direction
        if abs(roll) > 0.01:
            if (roll > 0 and corr_roll > 0) or (roll < 0 and corr_roll < 0):
                corr_roll *= -1.0
        if abs(pitch) > 0.01:
            if (pitch > 0 and corr_pitch > 0) or (pitch < 0 and corr_pitch < 0):
                corr_pitch *= -1.0
        
        correction = map_torso_correction_to_action(corr_roll, corr_pitch, baseline_action)
        
        # Verify direction is stabilizing
        if len(correction) >= 2:
            if abs(roll) > 0.01:
                roll_opposing = (roll > 0 and correction[0] < 0) or (roll < 0 and correction[0] > 0)
                if not roll_opposing:
                    correction[0] *= -1.0
            if abs(pitch) > 0.01:
                pitch_opposing = (pitch > 0 and correction[1] < 0) or (pitch < 0 and correction[1] > 0)
                if not pitch_opposing:
                    correction[1] *= -1.0
            
            if len(correction) > 4:
                correction[4:] = 0.0
        
        # ============================
        # PHASE-BASED TORQUE BLENDING
        # ============================
        
        # Compute prefall torque
        prefall_torque = correction.copy()
        
        PREFALL_MIN = 0.18
        PREFALL_MAX = 0.65
        prefall_gain = PREFALL_MIN + (PREFALL_MAX - PREFALL_MIN) * risk_ema
        prefall_gain = min(PREFALL_MAX, max(PREFALL_MIN, prefall_gain))
        
        if internal_zone in ("prefall", "fail"):
            prefall_torque *= prefall_gain
        else:
            prefall_torque *= 0.0
        
        # Compute safe torque
        catastrophic_risk = 1.0 if internal_zone == "fail" else (risk_ema * 1.2)
        catastrophic_risk = max(0.0, min(1.0, catastrophic_risk))
        
        safe_torque = np.zeros_like(baseline_action)
        if catastrophic_risk > 0.75:
            SAFE_GAIN = 0.12
            safe_torque[0] = -0.15 * roll * SAFE_GAIN
            safe_torque[1] = -0.15 * pitch * SAFE_GAIN
            if len(safe_torque) > 4:
                safe_torque[4:] = 0.0
        
        # Diagnostic tracking
        prefall_mag = np.linalg.norm(prefall_torque)
        if prefall_mag > 1e-6:
            core_state.prefall_magnitudes.append(prefall_mag)
            core_state.prefall_active_count += 1
        
        safe_mag = np.linalg.norm(safe_torque)
        if safe_mag > 1e-6:
            core_state.safe_magnitudes.append(safe_mag)
            core_state.safe_active_count += 1
        
        # Get phase-based blending weights
        w_prefall, w_safe = self.get_torque_weights(core_state.phase)
        
        # Blend torques
        combined_torque = w_prefall * prefall_torque + w_safe * safe_torque
        
        # Apply adaptive gain
        edon_delta = core_state.adaptive_gain * combined_torque
        
        # Get phase-dependent clamp ratio
        clamp_ratio = self.get_clamp_ratio(core_state.phase)
        
        # Clamp delta relative to baseline
        final_action = baseline_action + edon_delta
        final_action = clamp_action_relative_to_baseline(
            baseline_action,
            final_action,
            max_ratio=clamp_ratio
        )
        
        # Safety clamp
        final_action = clamp_action_relative_to_baseline(
            baseline_action=baseline_action,
            edon_action=final_action,
            max_ratio=clamp_ratio
        )
        
        # Final clip
        final_action = np.clip(final_action, -1.0, 1.0)
        
        # Return delta only
        delta = final_action - baseline_action
        delta_mag = np.linalg.norm(delta)
        core_state.delta_magnitudes.append(delta_mag)
        
        return delta


class LearnedEdonPolicy(EdonPolicyBase):
    """
    Learned EDON policy (v6).
    
    Uses a trained PyTorch MLP to predict edon_delta from features and state.
    Falls back to heuristic if model is not available.
    """
    
    def __init__(self, heuristic_fallback: Optional[EdonPolicyBase] = None, model_path: str = "models/edon_v6.pt"):
        """
        Initialize learned policy.
        
        Args:
            heuristic_fallback: Policy to use when model is not available
            model_path: Path to trained PyTorch model file
        """
        self.heuristic_fallback = heuristic_fallback
        self.model_path = model_path
        self.model = None
        self.input_size = None
        self.output_size = None
        self.device = None
        
        # Try to load PyTorch model
        self._load_model()
    
    def _load_model(self):
        """Load PyTorch model from disk."""
        try:
            import torch
            self.device = torch.device("cpu")  # Use CPU for inference
            
            model_file = Path(self.model_path)
            if not model_file.exists():
                print(f"[EDON-V6] Model file not found: {self.model_path}, using heuristic fallback")
                return
            
            # Load model checkpoint (weights_only=False for compatibility with numpy arrays in checkpoint)
            checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
            self.input_size = checkpoint['input_size']
            self.output_size = checkpoint['output_size']
            
            # Create model architecture
            class EdonV6MLP(torch.nn.Module):
                def __init__(self, input_size, output_size):
                    super().__init__()
                    self.fc1 = torch.nn.Linear(input_size, 128)
                    self.fc2 = torch.nn.Linear(128, 128)
                    self.fc3 = torch.nn.Linear(128, 64)
                    self.fc4 = torch.nn.Linear(64, output_size)
                    self.relu = torch.nn.ReLU()
                
                def forward(self, x):
                    x = self.relu(self.fc1(x))
                    x = self.relu(self.fc2(x))
                    x = self.relu(self.fc3(x))
                    x = self.fc4(x)
                    return x
            
            self.model = EdonV6MLP(self.input_size, self.output_size)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # Set to evaluation mode
            self.model.to(self.device)
            
            print(f"[EDON-V6] Loaded model from {self.model_path}")
            print(f"[EDON-V6] Input size: {self.input_size}, Output size: {self.output_size}")
            
        except ImportError:
            print("[EDON-V6] PyTorch not available, using heuristic fallback")
        except Exception as e:
            print(f"[EDON-V6] Failed to load model: {e}, using heuristic fallback")
    
    def _phase_to_one_hot(self, phase: str) -> np.ndarray:
        """Convert phase string to one-hot encoding."""
        phase_map = {"stable": 0, "warning": 1, "recovery": 2}
        one_hot = np.zeros(3, dtype=np.float32)
        if phase in phase_map:
            one_hot[phase_map[phase]] = 1.0
        return one_hot
    
    def _pack_input(
        self,
        features: EdonFeatures,
        core_state: EdonCoreState,
        baseline_action: np.ndarray
    ) -> np.ndarray:
        """
        Pack features, core_state, and baseline_action into input vector.
        
        Input vector:
        - tilt_mag (1)
        - vel_norm (1)
        - p_chaos (1)
        - p_stress (1)
        - risk_ema (1)
        - instability_score (1)
        - adaptive_gain (1)
        - phase one-hot (3)
        - baseline_action (variable)
        """
        input_vec = np.concatenate([
            [float(features.tilt_mag)],
            [float(features.vel_norm)],
            [float(features.p_chaos)],
            [float(features.p_stress)],
            [float(features.risk_ema)],
            [float(core_state.instability_score)],
            [float(core_state.adaptive_gain)],
            self._phase_to_one_hot(core_state.phase),
            baseline_action.astype(np.float32)
        ])
        return input_vec
    
    def _get_clamp_ratio(self, phase: str) -> float:
        """Get phase-dependent clamp ratio (reuse heuristic logic)."""
        if self.heuristic_fallback is not None and hasattr(self.heuristic_fallback, 'get_clamp_ratio'):
            return self.heuristic_fallback.get_clamp_ratio(phase)
        # Default clamp ratios if heuristic not available
        if phase == "stable":
            return 1.20
        elif phase == "warning":
            return 1.20
        else:  # recovery
            return 1.50
    
    def compute_delta(
        self,
        features: EdonFeatures,
        core_state: EdonCoreState,
        baseline_action: np.ndarray
    ) -> np.ndarray:
        """
        Compute EDON delta using learned model or fallback.
        
        If model is loaded, uses neural network inference and returns raw model output.
        Clamping is applied by EdonCore.compute_action(), not here.
        Otherwise, falls back to heuristic policy.
        """
        # Use model if available
        if self.model is not None:
            try:
                import torch
                
                # Pack input vector with EXACT ordering from training
                input_vec = np.concatenate([
                    [float(features.tilt_mag)],
                    [float(features.vel_norm)],
                    [float(features.p_chaos)],
                    [float(features.p_stress)],
                    [float(features.risk_ema)],
                    [float(core_state.instability_score)],
                    [float(core_state.adaptive_gain)],
                    self._phase_to_one_hot(core_state.phase),
                    baseline_action.astype(np.float32)
                ])
                
                # Check input size matches
                if len(input_vec) != self.input_size:
                    print(f"[EDON-V6] Warning: Input size mismatch. Expected {self.input_size}, got {len(input_vec)}")
                    if self.heuristic_fallback is not None:
                        return self.heuristic_fallback.compute_delta(features, core_state, baseline_action)
                    return np.zeros_like(baseline_action)
                
                # Convert to tensor and run inference
                with torch.no_grad():
                    inp = torch.from_numpy(input_vec.astype(np.float32)).unsqueeze(0).to(self.device)
                    out = self.model(inp)
                    delta = out.cpu().numpy()[0]  # Extract 1D array
                
                # Debug: Print delta magnitude to verify model is outputting non-zero values
                print("[EDON-V6-STEP] ||delta|| =", float(np.linalg.norm(delta)))
                
                # Ensure output size matches
                if len(delta) != len(baseline_action):
                    print(f"[EDON-V6] Warning: Output size mismatch. Expected {len(baseline_action)}, got {len(delta)}")
                    if self.heuristic_fallback is not None:
                        return self.heuristic_fallback.compute_delta(features, core_state, baseline_action)
                    return np.zeros_like(baseline_action)
                
                # Return raw model output (clamping will be done by EdonCore)
                return delta
                
            except Exception as e:
                print(f"[EDON-V6] Inference error: {e}, falling back to heuristic")
                import traceback
                traceback.print_exc()
                if self.heuristic_fallback is not None:
                    return self.heuristic_fallback.compute_delta(features, core_state, baseline_action)
                return np.zeros_like(baseline_action)
        
        # Fallback to heuristic
        if self.heuristic_fallback is not None:
            return self.heuristic_fallback.compute_delta(features, core_state, baseline_action)
        
        # No-op if no fallback
        return np.zeros_like(baseline_action)


class V7LearnedPolicy(EdonPolicyBase):
    """
    EDON v7 Learned Policy (PPO-trained).
    
    Uses a trained PyTorch MLP (trained with PPO) to predict edon_delta.
    Uses simplified observation packing compared to v6.
    """
    
    def __init__(self, heuristic_fallback: Optional[EdonPolicyBase] = None, model_path: str = "models/edon_v7.pt"):
        """
        Initialize v7 learned policy.
        
        Args:
            heuristic_fallback: Policy to use when model is not available
            model_path: Path to trained PyTorch model file
        """
        self.heuristic_fallback = heuristic_fallback
        self.model_path = model_path
        self.model = None
        self.input_size = None
        self.output_size = None
        self.device = None
        
        # Try to load PyTorch model
        self._load_model()
    
    def _load_model(self):
        """Load PyTorch model from disk."""
        try:
            import torch
            self.device = torch.device("cpu")  # Use CPU for inference
            
            model_file = Path(self.model_path)
            if not model_file.exists():
                print(f"[EDON-V7] Model file not found: {self.model_path}, using heuristic fallback")
                return
            
            # Load model checkpoint (weights_only=False for compatibility with numpy arrays in checkpoint)
            checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
            self.input_size = checkpoint['input_size']
            self.output_size = checkpoint['output_size']
            
            # Create model architecture (same as training - tanh-squashed policy)
            # Training uses nn.Sequential with layers named net.0, net.2, net.4, etc.
            class EdonV7MLP(torch.nn.Module):
                def __init__(self, input_size, output_size, max_delta=1.0, hidden_sizes=[128, 128, 64]):
                    super().__init__()
                    # Shared network (matches training: Sequential with Linear/ReLU pairs)
                    layers = []
                    prev_size = input_size
                    for hidden_size in hidden_sizes:
                        layers.append(torch.nn.Linear(prev_size, hidden_size))
                        layers.append(torch.nn.ReLU())
                        prev_size = hidden_size
                    self.net = torch.nn.Sequential(*layers)
                    
                    # Action head (mean)
                    self.mu_head = torch.nn.Linear(prev_size, output_size)
                    
                    # Learnable log standard deviation
                    self.log_std = torch.nn.Parameter(torch.zeros(output_size))
                    self.max_delta = max_delta
                
                def forward(self, x):
                    x = self.net(x)
                    mu = self.mu_head(x)
                    std = self.log_std.exp().clamp(min=1e-6, max=1.0)
                    return mu, std
                
                def sample_action(self, obs):
                    """Sample action with tanh squashing."""
                    mu, std = self.forward(obs)
                    dist = torch.distributions.Normal(mu, std)
                    raw_action = dist.rsample()
                    log_prob = dist.log_prob(raw_action).sum(dim=-1)
                    action = torch.tanh(raw_action) * self.max_delta
                    return action, log_prob
            
            max_delta = checkpoint.get('max_delta', 1.0)
            self.model = EdonV7MLP(self.input_size, self.output_size, max_delta=max_delta)
            self.model.load_state_dict(checkpoint['policy_state_dict'])
            self.model.eval()  # Set to evaluation mode
            self.model.to(self.device)
            
            print(f"[EDON-V7] Loaded model from {self.model_path}")
            print(f"[EDON-V7] Input size: {self.input_size}, Output size: {self.output_size}")
            
        except ImportError:
            print("[EDON-V7] PyTorch not available, using heuristic fallback")
        except Exception as e:
            print(f"[EDON-V7] Failed to load model: {e}, using heuristic fallback")
    
    def _pack_observation(self, obs: dict, baseline_action: np.ndarray) -> np.ndarray:
        """
        Pack observation into input vector (same format as training).
        
        Uses simplified features compared to v6:
        - roll, pitch, roll_velocity, pitch_velocity
        - com_x, com_y, com_velocity_x, com_velocity_y
        - tilt_mag, vel_norm, com_norm, com_vel_norm
        - baseline_action
        """
        # Extract key features
        roll = float(obs.get("roll", 0.0))
        pitch = float(obs.get("pitch", 0.0))
        roll_velocity = float(obs.get("roll_velocity", 0.0))
        pitch_velocity = float(obs.get("pitch_velocity", 0.0))
        com_x = float(obs.get("com_x", 0.0))
        com_y = float(obs.get("com_y", 0.0))
        com_velocity_x = float(obs.get("com_velocity_x", 0.0))
        com_velocity_y = float(obs.get("com_velocity_y", 0.0))
        
        # Compute derived features
        tilt_mag = np.sqrt(roll**2 + pitch**2)
        vel_norm = np.sqrt(roll_velocity**2 + pitch_velocity**2)
        com_norm = np.sqrt(com_x**2 + com_y**2)
        com_vel_norm = np.sqrt(com_velocity_x**2 + com_velocity_y**2)
        
        # Pack: features + baseline action
        input_vec = np.concatenate([
            [roll, pitch, roll_velocity, pitch_velocity],
            [com_x, com_y, com_velocity_x, com_velocity_y],
            [tilt_mag, vel_norm, com_norm, com_vel_norm],
            baseline_action.astype(np.float32)
        ])
        
        return input_vec
    
    def compute_delta(
        self,
        features: EdonFeatures,
        core_state: EdonCoreState,
        baseline_action: np.ndarray
    ) -> np.ndarray:
        """
        Compute EDON delta using v7 learned model.
        
        Note: v7 uses simplified observation packing, so we reconstruct obs from features.
        """
        if self.model is None:
            if self.heuristic_fallback is not None:
                return self.heuristic_fallback.compute_delta(features, core_state, baseline_action)
            return np.zeros_like(baseline_action)
        
        try:
            import torch
            
            # Reconstruct observation dict from features (simplified)
            # v7 training uses raw obs, so we approximate it
            obs_dict = {
                "roll": features.tilt_mag * 0.5,  # Approximate
                "pitch": features.tilt_mag * 0.5,
                "roll_velocity": features.vel_norm * 0.5,
                "pitch_velocity": features.vel_norm * 0.5,
                "com_x": 0.0,  # Not available in features
                "com_y": 0.0,
                "com_velocity_x": 0.0,
                "com_velocity_y": 0.0,
            }
            
            # Pack observation
            input_vec = self._pack_observation(obs_dict, baseline_action)
            
            # Check input size
            if len(input_vec) != self.input_size:
                print(f"[EDON-V7] Warning: Input size mismatch. Expected {self.input_size}, got {len(input_vec)}")
                if self.heuristic_fallback is not None:
                    return self.heuristic_fallback.compute_delta(features, core_state, baseline_action)
                return np.zeros_like(baseline_action)
            
            # Convert to tensor and run inference (use sample_action for tanh-squashed output)
            with torch.no_grad():
                inp = torch.from_numpy(input_vec.astype(np.float32)).unsqueeze(0).to(self.device)
                action_tensor, _ = self.model.sample_action(inp)
                delta = action_tensor.cpu().numpy()[0]  # Extract 1D array (already tanh-squashed)
            
            return delta
            
        except Exception as e:
            print(f"[EDON-V7] Inference error: {e}")
            if self.heuristic_fallback is not None:
                return self.heuristic_fallback.compute_delta(features, core_state, baseline_action)
            return np.zeros_like(baseline_action)


class EdonCore:
    """
    EDON v5 Core orchestrator.
    
    Manages feature extraction, state updates, and policy execution.
    This is the single entry point for EDON v5 architecture.
    """
    
    def __init__(self, base_gain: float, policy: EdonPolicyBase, train_logger: Optional[EdonTrainLogger] = None):
        """
        Initialize EDON core.
        
        Args:
            base_gain: Base EDON gain (from CLI)
            policy: Policy implementation (e.g., HeuristicEdonPolicy)
            train_logger: Optional training logger for data collection
        """
        self.base_gain = base_gain
        self.policy = policy
        self.train_logger = train_logger
        self.state = EdonCoreState()
        
        # Legacy controller state (for risk_ema, internal_zone)
        self.controller_state = {
            'risk_ema': 0.0,
            'internal_zone': 'safe',
            'episode_id': -1
        }
        
        # Track step ID for logging
        self.step_id = 0
        self.current_episode_id = -1
        self.profile = "unknown"  # Will be set from args
        self.mode = "edon"
        
        # Phase transition thresholds (same as v4)
        self.T_WARNING_ON = 0.42
        self.T_WARNING_OFF = 0.3
        self.T_RECOVERY_ON = 0.58
        self.T_RECOVERY_OFF = 0.30
        
        # Smoothing parameters (same as v4)
        self.ALPHA_TILT = 0.2
        self.ALPHA_VEL = 0.2
        self.ALPHA_GAIN = 0.65
        self.ALPHA_INSTABILITY = 0.15
        
        # Logging
        self.logging_enabled = True
        self._logged_config = False
        self._logged_gain_usage = False
        
        # Fail-risk model (lazy-loaded for v8)
        self._fail_risk_model = None
        self._fail_risk_model_path = "models/edon_fail_risk_v1.pt"
    
    def _load_fail_risk_model(self):
        """Lazy-load fail-risk model if available."""
        if self._fail_risk_model is not None:
            return
        
        try:
            import torch
            from training.fail_risk_model import FailRiskModel
            
            model_path = Path(self._fail_risk_model_path)
            if not model_path.exists():
                return  # Model not available, fail-risk will be 0.0
            
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            input_size = checkpoint.get("input_size", 15)
            model = FailRiskModel(input_size=input_size)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self._fail_risk_model = model
        except Exception as e:
            # Fail silently, fail-risk will default to 0.0
            pass
    
    def _compute_fail_risk(self, obs: dict, features: EdonFeatures) -> float:
        """
        Compute fail-risk using fail-risk model.
        
        Args:
            obs: Current observation
            features: Extracted features
        
        Returns:
            fail_risk ∈ [0, 1]
        """
        # Lazy-load model
        self._load_fail_risk_model()
        
        if self._fail_risk_model is None:
            return 0.0
        
        try:
            import torch
            
            # Pack feature vector (same as fail_risk_model.extract_features_from_step)
            roll = float(obs.get("roll", 0.0))
            pitch = float(obs.get("pitch", 0.0))
            roll_velocity = float(obs.get("roll_velocity", 0.0))
            pitch_velocity = float(obs.get("pitch_velocity", 0.0))
            com_x = float(obs.get("com_x", 0.0))
            com_y = float(obs.get("com_y", 0.0))
            com_velocity_x = float(obs.get("com_velocity_x", 0.0))
            com_velocity_y = float(obs.get("com_velocity_y", 0.0))
            
            tilt_mag = features.tilt_mag
            vel_norm = features.vel_norm
            com_norm = np.sqrt(com_x**2 + com_y**2)
            com_vel_norm = np.sqrt(com_velocity_x**2 + com_velocity_y**2)
            
            instability_score = features.risk_ema  # Use risk_ema as proxy
            risk_ema = features.risk_ema
            
            # Phase encoding
            phase_map = {"stable": 0.0, "warning": 1.0, "recovery": 2.0, "prefall": 3.0, "fail": 4.0}
            phase_encoded = phase_map.get(self.state.phase, 0.0)
            
            # Pack feature vector
            feature_vec = np.array([
                roll, pitch, roll_velocity, pitch_velocity,
                com_x, com_y, com_velocity_x, com_velocity_y,
                tilt_mag, vel_norm, com_norm, com_vel_norm,
                instability_score, risk_ema, phase_encoded
            ], dtype=np.float32)
            
            # Run inference
            with torch.no_grad():
                feature_tensor = torch.FloatTensor(feature_vec).unsqueeze(0)
                fail_risk = self._fail_risk_model(feature_tensor).item()
            
            return float(fail_risk)
        except Exception:
            return 0.0
    
    def extract_features(
        self,
        obs: dict,
        edon_state_raw: Optional[Dict[str, Any]],
        controller_state: Optional[Dict[str, Any]] = None
    ) -> EdonFeatures:
        """
        Extract features from observation and EDON state.
        
        This moves the feature extraction logic from apply_edon_regulation.
        """
        # Extract tilt and velocities
        roll = obs.get("roll", 0.0)
        pitch = obs.get("pitch", 0.0)
        roll_velocity = obs.get("roll_velocity", 0.0)
        pitch_velocity = obs.get("pitch_velocity", 0.0)
        
        # Compute tilt magnitude and velocity norm
        tilt_magnitude = np.sqrt(roll**2 + pitch**2)
        vel_norm = np.sqrt(roll_velocity**2 + pitch_velocity**2)
        
        # Classify tilt zone
        max_tilt = max(abs(roll), abs(pitch))
        if max_tilt > config.FAIL_LIMIT:
            tilt_zone = "fail"
        elif max_tilt > config.PREFALL_LIMIT or tilt_magnitude > config.PREFALL_LIMIT:
            tilt_zone = "prefall"
        elif max_tilt > config.SAFE_LIMIT or tilt_magnitude > config.SAFE_LIMIT:
            tilt_zone = "prefall"
        else:
            tilt_zone = "safe"
        
        # Use internal controller state if not provided
        if controller_state is None:
            controller_state = self.controller_state
        
        # Get risk score and update EMA
        risk_score = 0.0
        if edon_state_raw is not None:
            risk_score = compute_risk_score(edon_state_raw)
            alpha = 0.15
            controller_state['risk_ema'] = alpha * risk_score + (1 - alpha) * controller_state.get('risk_ema', 0.0)
        else:
            controller_state['risk_ema'] = 0.0
        
        risk_ema = controller_state['risk_ema']
        
        # Determine internal zone
        internal_zone = controller_state.get('internal_zone', tilt_zone)
        if tilt_zone == "safe" and risk_ema >= 0.5 and max_tilt > 0.6 * config.SAFE_LIMIT:
            internal_zone = "prefall"
        elif tilt_zone in ("prefall", "fail"):
            internal_zone = tilt_zone
        elif internal_zone == "prefall":
            if max_tilt < config.SAFE_LIMIT * 0.6 and risk_ema < 0.3:
                internal_zone = "safe"
        else:
            internal_zone = "safe"
        
        controller_state['internal_zone'] = internal_zone
        
        # Extract EDON signals
        p_chaos = 0.0
        p_stress = 0.0
        if edon_state_raw is not None:
            p_chaos = edon_state_raw.get("p_chaos", 0.0)
            p_stress = edon_state_raw.get("p_stress", 0.0)
            if not isinstance(p_chaos, (int, float)):
                p_chaos = 0.0
            if not isinstance(p_stress, (int, float)):
                p_stress = 0.0
            p_chaos = max(0.0, min(1.0, p_chaos))
            p_stress = max(0.0, min(1.0, p_stress))
        
        return EdonFeatures(
            tilt_mag=tilt_magnitude,
            vel_norm=vel_norm,
            p_chaos=p_chaos,
            p_stress=p_stress,
            risk_ema=risk_ema,
            tilt_zone=tilt_zone,
            internal_zone=internal_zone,
            roll=roll,
            pitch=pitch,
            roll_velocity=roll_velocity,
            pitch_velocity=pitch_velocity,
            max_tilt=max_tilt,
            risk_score=risk_score
        )
    
    def compute_instability_score(self, features: EdonFeatures) -> float:
        """
        Compute instability score from features.
        
        This is the same logic as EDONController.compute_instability_score.
        """
        # Normalize tilt (0-1 scale, assuming max reasonable tilt ~0.4 rad)
        tilt_normalized = min(features.tilt_mag / 0.4, 1.0)
        
        # Normalize velocity (0-1 scale, assuming max reasonable velocity ~5.0 rad/s)
        vel_normalized = min(features.vel_norm / 5.0, 1.0)
        
        # Combine signals with weights
        instability = (
            0.35 * tilt_normalized +
            0.25 * vel_normalized +
            0.20 * features.p_chaos +
            0.15 * features.p_stress +
            0.05 * features.risk_ema
        )
        
        # Boost if in prefall/fail zone
        if features.tilt_zone in ("prefall", "fail"):
            instability = min(1.0, instability * 1.15)
        
        return max(0.0, min(1.0, instability))
    
    def update_phase(self, instability_score: float) -> str:
        """
        Update phase based on instability score with hysteresis.
        
        Same logic as EDONController.update_phase.
        """
        current_phase = self.state.phase
        
        if current_phase == "stable":
            if instability_score > self.T_WARNING_ON:
                new_phase = "warning"
            else:
                new_phase = "stable"
        elif current_phase == "warning":
            if instability_score > self.T_RECOVERY_ON:
                new_phase = "recovery"
            elif instability_score < self.T_WARNING_OFF:
                new_phase = "stable"
            else:
                new_phase = "warning"
        else:  # recovery
            if instability_score < self.T_RECOVERY_OFF:
                new_phase = "warning"
            else:
                new_phase = "recovery"
        
        # Update phase counts
        if new_phase != current_phase:
            self.state.phase = new_phase
        self.state.phase_counts[new_phase] = self.state.phase_counts.get(new_phase, 0) + 1
        
        return new_phase
    
    def compute_adaptive_gain(self) -> float:
        """
        Compute adaptive gain based on current phase.
        
        Same logic as EDONController.compute_adaptive_gain, but uses policy's gain values.
        """
        phase = self.state.phase
        
        # Get phase-based gain multiplier from policy
        if isinstance(self.policy, HeuristicEdonPolicy):
            if phase == "stable":
                phase_gain = self.policy.gain_stable
            elif phase == "warning":
                phase_gain = self.policy.gain_warning
            else:  # recovery
                phase_gain = self.policy.gain_recovery
        else:
            # Fallback for other policies
            phase_gain = 1.0
        
        # Compute raw adaptive gain
        adaptive_gain = self.base_gain * phase_gain
        
        # Cap recovery gain
        if phase == "recovery":
            adaptive_gain = min(adaptive_gain, 1.1)  # GAIN_RECOVERY_MAX
        
        # Smooth with previous gain
        last_gain = self.state.last_gain
        if last_gain > 0.0:
            adaptive_gain = self.ALPHA_GAIN * adaptive_gain + (1 - self.ALPHA_GAIN) * last_gain
        
        # Store for next iteration
        self.state.last_gain = adaptive_gain
        self.state.adaptive_gain = adaptive_gain
        
        return adaptive_gain
    
    def update_state(self, features: EdonFeatures) -> None:
        """
        Update core state based on features.
        
        This includes:
        - Computing instability score
        - Updating smoothed signals
        - Updating phase
        - Computing adaptive gain
        """
        # Compute instability score
        instability_score = self.compute_instability_score(features)
        
        # Store raw instability score in state (needed for learned policy input)
        self.state.instability_score = instability_score
        
        # Update smoothed signals
        self.state.smoothed_tilt = (
            self.ALPHA_TILT * features.tilt_mag +
            (1 - self.ALPHA_TILT) * self.state.smoothed_tilt
        )
        self.state.smoothed_vel_norm = (
            self.ALPHA_VEL * features.vel_norm +
            (1 - self.ALPHA_VEL) * self.state.smoothed_vel_norm
        )
        self.state.smoothed_instability = (
            self.ALPHA_INSTABILITY * instability_score +
            (1 - self.ALPHA_INSTABILITY) * self.state.smoothed_instability
        )
        
        # Update phase with hysteresis (use smoothed instability)
        self.update_phase(self.state.smoothed_instability)
        
        # Compute adaptive gain
        self.compute_adaptive_gain()
        
        # Track for logging
        self.state.instability_history.append(instability_score)
        self.state.gain_history.append(self.state.adaptive_gain)
        self.state.step_count += 1
    
    def compute_action(
        self,
        baseline_action: np.ndarray,
        obs: dict,
        edon_state_raw: Optional[Dict[str, Any]],
        controller_state: Optional[Dict[str, Any]] = None,
        done: bool = False,
        interventions_so_far: int = 0
    ) -> np.ndarray:
        """
        Compute final action using EDON v5 pipeline.
        
        This is the main entry point for EDON v5:
        1. Extract features
        2. Update state
        3. Compute delta via policy
        4. Log training data (if enabled)
        5. Return final action
        
        Args:
            baseline_action: Baseline controller action
            obs: Observation dictionary
            edon_state_raw: EDON state from API
            controller_state: Legacy controller state dict (optional, uses internal if None)
            done: Whether episode is done (for logging)
            interventions_so_far: Number of interventions so far (for logging)
            
        Returns:
            Final action (baseline + EDON delta)
        """
        # Use internal controller state if not provided
        if controller_state is None:
            controller_state = self.controller_state
        
        # Extract features
        features = self.extract_features(obs, edon_state_raw, controller_state)
        
        # Update state
        self.update_state(features)
        
        # Compute fail-risk if model is available (for v8 and logging)
        fail_risk = self._compute_fail_risk(obs, features)
        self.state.fail_risk = fail_risk  # Store in state for logging
        
        # Compute delta via policy
        delta = self.policy.compute_delta(features, self.state, baseline_action)
        
        # Track delta magnitude for diagnostics (for both v5_heuristic and v6_learned)
        delta_mag = np.linalg.norm(delta)
        self.state.delta_magnitudes.append(delta_mag)
        
        # Compute final action
        final_action = baseline_action + delta
        
        # Apply phase-dependent clamping (same logic for both v5_heuristic and v6_learned)
        if isinstance(self.policy, LearnedEdonPolicy):
            # For v6_learned, get clamp ratio from heuristic fallback or default
            if self.policy.heuristic_fallback is not None and hasattr(self.policy.heuristic_fallback, 'get_clamp_ratio'):
                clamp_ratio = self.policy.heuristic_fallback.get_clamp_ratio(self.state.phase)
            else:
                # Default clamp ratios
                if self.state.phase == "stable":
                    clamp_ratio = 1.20
                elif self.state.phase == "warning":
                    clamp_ratio = 1.20
                else:  # recovery
                    clamp_ratio = 1.50
        elif isinstance(self.policy, HeuristicEdonPolicy):
            clamp_ratio = self.policy.get_clamp_ratio(self.state.phase)
        else:
            clamp_ratio = 1.20  # Default fallback
        
        # Clamp to prevent overshoot (phase-dependent)
        final_action = clamp_action_relative_to_baseline(
            baseline_action,
            final_action,
            max_ratio=clamp_ratio
        )
        
        # Safety clamp (phase-dependent)
        final_action = clamp_action_relative_to_baseline(
            baseline_action=baseline_action,
            edon_action=final_action,
            max_ratio=clamp_ratio
        )
        
        # Final clip
        final_action = np.clip(final_action, -1.0, 1.0)
        
        # Log training data if logger is enabled
        if self.train_logger is not None:
            step_record = {
                "episode_id": self.current_episode_id,
                "step_id": self.step_id,
                "profile": self.profile,
                "mode": self.mode,
                "features": {
                    "tilt_mag": float(features.tilt_mag),
                    "vel_norm": float(features.vel_norm),
                    "p_chaos": float(features.p_chaos),
                    "p_stress": float(features.p_stress),
                    "risk_ema": float(features.risk_ema),
                    "tilt_zone": features.tilt_zone,
                },
                "core_state": {
                    "phase": self.state.phase,
                    "instability_score": float(self.state.instability_score),
                    "adaptive_gain": float(self.state.adaptive_gain),
                    "fail_risk": float(self.state.fail_risk),
                },
                "fail_risk": float(self.state.fail_risk),
                "baseline_action": baseline_action.tolist(),
                "edon_delta": delta.tolist(),
                "final_action": final_action.tolist(),
                "done": done,
                "interventions_so_far": interventions_so_far,
            }
            self.train_logger.log_step(step_record)
        
        # Increment step counter
        self.step_id += 1
        
        # Log gain usage on first call
        if not self._logged_gain_usage:
            if isinstance(self.policy, LearnedEdonPolicy):
                print(f"[EDON-V6] base_gain={self.base_gain:.3f} adaptive_gain={self.state.adaptive_gain:.3f} phase={self.state.phase}")
            else:
                print(f"[EDON-V5] base_gain={self.base_gain:.3f} adaptive_gain={self.state.adaptive_gain:.3f} phase={self.state.phase}")
            self._logged_gain_usage = True
            controller_state["_logged_gain_usage"] = True
        
        return final_action
    
    def log_episode_summary(self, episode_id: int, interventions: int = 0, stability_score: float = 0.0):
        """
        Log adaptive behavior summary for an episode.
        
        Args:
            episode_id: Episode ID
            interventions: Number of interventions in this episode
            stability_score: Stability score for this episode
        """
        if not self.logging_enabled:
            return
        
        # Log config on first episode
        if not self._logged_config and episode_id == 0:
            # Determine architecture label
            if isinstance(self.policy, V7LearnedPolicy):
                arch_label = "V7"
            elif isinstance(self.policy, LearnedEdonPolicy):
                # Check which model path is being used
                if hasattr(self.policy, "model_path") and "v6_1" in self.policy.model_path:
                    arch_label = "V6.1"
                else:
                    arch_label = "V6"
            else:
                arch_label = "V5"
            print(f"[EDON-{arch_label}] Configuration:")
            print(f"  base_gain={self.base_gain:.3f}")
            print(f"  thresholds: warning_on={self.T_WARNING_ON:.2f}, warning_off={self.T_WARNING_OFF:.2f}")
            print(f"  thresholds: recovery_on={self.T_RECOVERY_ON:.2f}, recovery_off={self.T_RECOVERY_OFF:.2f}")
            self._logged_config = True
        
        # Compute episode averages
        if len(self.state.instability_history) > 0:
            avg_instability = np.mean(self.state.instability_history)
        else:
            avg_instability = 0.0
        
        if len(self.state.gain_history) > 0:
            avg_gain = np.mean(self.state.gain_history)
        else:
            avg_gain = self.base_gain
        
        # Compute torque diagnostics
        total_steps = self.state.step_count
        if len(self.state.prefall_magnitudes) > 0:
            avg_prefall = np.mean(self.state.prefall_magnitudes)
        else:
            avg_prefall = 0.0
        
        if len(self.state.safe_magnitudes) > 0:
            avg_safe = np.mean(self.state.safe_magnitudes)
        else:
            avg_safe = 0.0
        
        if len(self.state.delta_magnitudes) > 0:
            avg_delta = np.mean(self.state.delta_magnitudes)
        else:
            avg_delta = 0.0
        
        # Compute activity percentages
        prefall_active_pct = (self.state.prefall_active_count / total_steps * 100.0) if total_steps > 0 else 0.0
        safe_active_pct = (self.state.safe_active_count / total_steps * 100.0) if total_steps > 0 else 0.0
        
        # Log summary with diagnostics
        phase_counts_str = ", ".join([f"{k}={v}" for k, v in self.state.phase_counts.items()])
        # Determine architecture label
        if isinstance(self.policy, LearnedEdonPolicy):
            # Check which model path is being used
            if hasattr(self.policy, "model_path") and "v6_1" in self.policy.model_path:
                arch_label = "V6.1"
            else:
                arch_label = "V6"
        else:
            arch_label = "V5"
        
        if isinstance(self.policy, LearnedEdonPolicy):
            # For learned policies, show policy vs baseline usage
            policy_steps = getattr(self, "_episode_policy_steps", 0)
            baseline_steps = getattr(self, "_episode_baseline_steps", 0)
            total_steps = policy_steps + baseline_steps
            policy_pct = (policy_steps / total_steps * 100.0) if total_steps > 0 else 0.0
            
            print(
                f"[EDON-{arch_label}] episode={episode_id} "
                f"phase_counts=({phase_counts_str}) "
                f"avg_instability={avg_instability:.3f} "
                f"avg_gain={avg_gain:.3f} "
                f"avg_delta={avg_delta:.4f} "
                f"policy_steps={policy_steps} baseline_steps={baseline_steps} (policy={policy_pct:.1f}%)"
            )
        else:
            # For v5_heuristic, show all diagnostics
            print(
                f"[EDON-{arch_label}] episode={episode_id} "
                f"phase_counts=({phase_counts_str}) "
                f"avg_instability={avg_instability:.3f} "
                f"avg_gain={avg_gain:.3f} "
                f"avg_prefall={avg_prefall:.4f} "
                f"avg_safe={avg_safe:.4f} "
                f"avg_delta={avg_delta:.4f} "
                f"prefall_active={prefall_active_pct:.1f}% "
                f"safe_active={safe_active_pct:.1f}%"
            )
        
        # Log episode summary to training logger if enabled
        if self.train_logger is not None:
            episode_summary = {
                "episode_id": episode_id,
                "phase_counts": dict(self.state.phase_counts),
                "avg_instability": float(avg_instability),
                "avg_gain": float(avg_gain),
                "interventions": interventions,
                "stability_score": float(stability_score),
                "total_steps": total_steps,
            }
            self.train_logger.log_episode_summary(episode_summary)
        
        # Reset episode tracking
        self.state.phase_counts = {"stable": 0, "warning": 0, "recovery": 0}
        self.state.instability_history = []
        self.state.gain_history = []
        self.state.prefall_magnitudes = []
        self.state.safe_magnitudes = []
        self.state.delta_magnitudes = []
        self.state.prefall_active_count = 0
        self.state.safe_active_count = 0
    
    def reset_episode(self, episode_id: int):
        """Reset state for new episode."""
        self.current_episode_id = episode_id
        self.step_id = 0
        self.state.episode_id = episode_id
        self.state.step_count = 0
        # Keep smoothed signals (they provide continuity)
        # Reset phase to stable
        self.state.phase = "stable"
        self.state.phase_counts = {"stable": 0, "warning": 0, "recovery": 0}
        # Reset diagnostic tracking
        self.state.prefall_magnitudes = []
        self.state.safe_magnitudes = []
        self.state.delta_magnitudes = []
        self.state.prefall_active_count = 0
        self.state.safe_active_count = 0
        # Reset controller state
        self.controller_state['episode_id'] = episode_id
        self.controller_state['risk_ema'] = 0.0
        self.controller_state['internal_zone'] = 'safe'


# Global EDON controller instance (created per evaluation run) - v4 legacy
_edon_controller_instance: Optional[EDONController] = None

# Global EDON v5 core instance
_edon_core_instance: Optional[EdonCore] = None
_edon_arch: Optional[str] = None  # Architecture selection ("v5_heuristic", etc.)


def get_edon_core(
    base_gain: float,
    edon_arch: str = "v5_heuristic",
    train_logger: Optional[EdonTrainLogger] = None,
    profile: str = "unknown",
    mode: str = "edon"
) -> EdonCore:
    """
    Get or create EDON v5 core instance.
    
    Args:
        base_gain: Base EDON gain from CLI
        edon_arch: Architecture/policy selection
        train_logger: Optional training logger
        profile: Stress profile name (for logging)
        mode: Mode name (for logging)
        
    Returns:
        EdonCore instance
    """
    global _edon_core_instance, _edon_arch
    
    # Create new instance if architecture changed or doesn't exist
    if _edon_core_instance is None or _edon_arch != edon_arch:
        # Create policy based on architecture
        if edon_arch == "v5_heuristic":
            # Use config values from _EDON_CONFIG
            policy = HeuristicEdonPolicy(
                gain_stable=_EDON_CONFIG["GAIN_STABLE"],
                gain_warning=_EDON_CONFIG["GAIN_WARNING"],
                gain_recovery=_EDON_CONFIG["GAIN_RECOVERY"],
                clamp_ratio_stable=_EDON_CONFIG["CLAMP_RATIO_STABLE"],
                clamp_ratio_warning=_EDON_CONFIG["CLAMP_RATIO_WARNING"],
                clamp_ratio_recovery=_EDON_CONFIG["CLAMP_RATIO_RECOVERY"],
                w_prefall_stable=_EDON_CONFIG["W_PREFALL_STABLE"],
                w_safe_stable=_EDON_CONFIG["W_SAFE_STABLE"],
                w_prefall_warning=_EDON_CONFIG["W_PREFALL_WARNING"],
                w_safe_warning=_EDON_CONFIG["W_SAFE_WARNING"],
                w_prefall_recovery=_EDON_CONFIG["W_PREFALL_RECOVERY"],
                w_safe_recovery=_EDON_CONFIG["W_SAFE_RECOVERY"],
            )
        elif edon_arch == "v6_learned":
            # Create heuristic policy as fallback
            heuristic = HeuristicEdonPolicy(
                gain_stable=_EDON_CONFIG["GAIN_STABLE"],
                gain_warning=_EDON_CONFIG["GAIN_WARNING"],
                gain_recovery=_EDON_CONFIG["GAIN_RECOVERY"],
                clamp_ratio_stable=_EDON_CONFIG["CLAMP_RATIO_STABLE"],
                clamp_ratio_warning=_EDON_CONFIG["CLAMP_RATIO_WARNING"],
                clamp_ratio_recovery=_EDON_CONFIG["CLAMP_RATIO_RECOVERY"],
                w_prefall_stable=_EDON_CONFIG["W_PREFALL_STABLE"],
                w_safe_stable=_EDON_CONFIG["W_SAFE_STABLE"],
                w_prefall_warning=_EDON_CONFIG["W_PREFALL_WARNING"],
                w_safe_warning=_EDON_CONFIG["W_SAFE_WARNING"],
                w_prefall_recovery=_EDON_CONFIG["W_PREFALL_RECOVERY"],
                w_safe_recovery=_EDON_CONFIG["W_SAFE_RECOVERY"],
            )
            # Learned policy delegates to heuristic for now
            policy = LearnedEdonPolicy(heuristic_fallback=heuristic, model_path="models/edon_v6.pt")
        elif edon_arch == "v6_1_learned":
            # Create heuristic policy as fallback
            heuristic = HeuristicEdonPolicy(
                gain_stable=_EDON_CONFIG["GAIN_STABLE"],
                gain_warning=_EDON_CONFIG["GAIN_WARNING"],
                gain_recovery=_EDON_CONFIG["GAIN_RECOVERY"],
                clamp_ratio_stable=_EDON_CONFIG["CLAMP_RATIO_STABLE"],
                clamp_ratio_warning=_EDON_CONFIG["CLAMP_RATIO_WARNING"],
                clamp_ratio_recovery=_EDON_CONFIG["CLAMP_RATIO_RECOVERY"],
                w_prefall_stable=_EDON_CONFIG["W_PREFALL_STABLE"],
                w_safe_stable=_EDON_CONFIG["W_SAFE_STABLE"],
                w_prefall_warning=_EDON_CONFIG["W_PREFALL_WARNING"],
                w_safe_warning=_EDON_CONFIG["W_SAFE_WARNING"],
                w_prefall_recovery=_EDON_CONFIG["W_PREFALL_RECOVERY"],
                w_safe_recovery=_EDON_CONFIG["W_SAFE_RECOVERY"],
            )
            # v6.1 learned policy uses new model
            policy = LearnedEdonPolicy(heuristic_fallback=heuristic, model_path="models/edon_v6_1.pt")
        elif edon_arch == "v7_learned":
            # Create heuristic policy as fallback
            heuristic = HeuristicEdonPolicy(
                gain_stable=_EDON_CONFIG["GAIN_STABLE"],
                gain_warning=_EDON_CONFIG["GAIN_WARNING"],
                gain_recovery=_EDON_CONFIG["GAIN_RECOVERY"],
                clamp_ratio_stable=_EDON_CONFIG["CLAMP_RATIO_STABLE"],
                clamp_ratio_warning=_EDON_CONFIG["CLAMP_RATIO_WARNING"],
                clamp_ratio_recovery=_EDON_CONFIG["CLAMP_RATIO_RECOVERY"],
                w_prefall_stable=_EDON_CONFIG["W_PREFALL_STABLE"],
                w_safe_stable=_EDON_CONFIG["W_SAFE_STABLE"],
                w_prefall_warning=_EDON_CONFIG["W_PREFALL_WARNING"],
                w_safe_warning=_EDON_CONFIG["W_SAFE_WARNING"],
                w_prefall_recovery=_EDON_CONFIG["W_PREFALL_RECOVERY"],
                w_safe_recovery=_EDON_CONFIG["W_SAFE_RECOVERY"],
            )
            # v7 learned policy (PPO-trained)
            policy = V7LearnedPolicy(heuristic_fallback=heuristic, model_path="models/edon_v7.pt")
        elif edon_arch == "v8_strategy":
            # v8 uses layered control with environment wrapper
            # For now, use heuristic as placeholder (v8 requires special evaluation loop)
            # TODO: Full v8 integration requires using EdonHumanoidEnvV8 in evaluation loop
            print("[EDON-V8] Note: v8_strategy requires special evaluation handling.")
            print("[EDON-V8] Using heuristic policy as placeholder. See training/V8_README.md for v8 evaluation.")
            policy = HeuristicEdonPolicy(
                gain_stable=_EDON_CONFIG["GAIN_STABLE"],
                gain_warning=_EDON_CONFIG["GAIN_WARNING"],
                gain_recovery=_EDON_CONFIG["GAIN_RECOVERY"],
                clamp_ratio_stable=_EDON_CONFIG["CLAMP_RATIO_STABLE"],
                clamp_ratio_warning=_EDON_CONFIG["CLAMP_RATIO_WARNING"],
                clamp_ratio_recovery=_EDON_CONFIG["CLAMP_RATIO_RECOVERY"],
                w_prefall_stable=_EDON_CONFIG["W_PREFALL_STABLE"],
                w_safe_stable=_EDON_CONFIG["W_SAFE_STABLE"],
                w_prefall_warning=_EDON_CONFIG["W_PREFALL_WARNING"],
                w_safe_warning=_EDON_CONFIG["W_SAFE_WARNING"],
                w_prefall_recovery=_EDON_CONFIG["W_PREFALL_RECOVERY"],
                w_safe_recovery=_EDON_CONFIG["W_SAFE_RECOVERY"],
            )
        else:
            raise ValueError(f"Unknown EDON architecture: {edon_arch}")
        
        _edon_core_instance = EdonCore(base_gain=base_gain, policy=policy, train_logger=train_logger)
        _edon_core_instance.profile = profile
        _edon_core_instance.mode = mode
        _edon_arch = edon_arch
    else:
        # Update base gain if changed
        _edon_core_instance.base_gain = base_gain
        if train_logger is not None:
            _edon_core_instance.train_logger = train_logger
        _edon_core_instance.profile = profile
        _edon_core_instance.mode = mode
    
    return _edon_core_instance


def map_torso_correction_to_action(
    corr_roll: float,
    corr_pitch: float,
    baseline_action: np.ndarray
) -> np.ndarray:
    """Map torso corrections to action space."""
    correction = np.zeros_like(baseline_action)
    if len(baseline_action) >= 4:
        correction[0] = corr_roll
        correction[1] = corr_pitch
        correction[2] = corr_roll * 0.4  # Slightly increased from 0.3
        correction[3] = corr_pitch * 0.4  # Slightly increased from 0.3
    return correction


def apply_forward_speed_scale(action: np.ndarray, speed_scale: float) -> np.ndarray:
    """Apply speed scaling to forward velocity component."""
    scaled_action = action.copy()
    if len(scaled_action) > 4:
        scaled_action[4:] *= speed_scale
    return scaled_action


def clamp_action_relative_to_baseline(
    baseline_action: np.ndarray,
    edon_action: np.ndarray,
    max_ratio: float = 1.2  # Default fallback (will be overridden by phase-dependent ratio)
) -> np.ndarray:
    """Clamp EDON action to prevent exceeding baseline by too much."""
    baseline_mag = np.linalg.norm(baseline_action)
    if baseline_mag < 0.01:
        edon_mag = np.linalg.norm(edon_action)
        if edon_mag > 0.2:
            scale = 0.2 / edon_mag
            return edon_action * scale
        return edon_action
    
    edon_mag = np.linalg.norm(edon_action)
    ratio = edon_mag / baseline_mag
    
    if ratio > max_ratio:
        scale = max_ratio / ratio
        clamped_action = baseline_action + (edon_action - baseline_action) * scale
        return clamped_action
    
    return edon_action


def apply_edon_regulation(
    baseline_action: np.ndarray,
    edon_state_raw: Optional[Dict[str, Any]],
    obs: dict,
    edon_gain: float,
    dt: float = 0.01,
    use_state: bool = True,
    controller_state: Optional[Dict[str, Any]] = None
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply adaptive EDON regulation to baseline action (V4 Adaptive).
    
    Uses stateful, context-aware adaptive controller with phase-based logic.
    """
    global _edon_controller_instance
    
    # Get or create EDON controller instance
    if _edon_controller_instance is None:
        _edon_controller_instance = EDONController(base_edon_gain=edon_gain)
    elif _edon_controller_instance.base_edon_gain != edon_gain:
        # Update base gain if it changed
        _edon_controller_instance.base_edon_gain = edon_gain
    
    controller = _edon_controller_instance
    
    # Extract tilt and tilt rate
    roll = obs.get("roll", 0.0)
    pitch = obs.get("pitch", 0.0)
    roll_velocity = obs.get("roll_velocity", 0.0)
    pitch_velocity = obs.get("pitch_velocity", 0.0)
    
    # Compute tilt magnitude and velocity norm
    tilt_magnitude = np.sqrt(roll**2 + pitch**2)
    vel_norm = np.sqrt(roll_velocity**2 + pitch_velocity**2)
    
    # Classify tilt zone
    max_tilt = max(abs(roll), abs(pitch))
    if max_tilt > config.FAIL_LIMIT:
        tilt_zone = "fail"
    elif max_tilt > config.PREFALL_LIMIT or tilt_magnitude > config.PREFALL_LIMIT:
        tilt_zone = "prefall"
    elif max_tilt > config.SAFE_LIMIT or tilt_magnitude > config.SAFE_LIMIT:
        tilt_zone = "prefall"
    else:
        tilt_zone = "safe"
    
    # Initialize controller state (legacy compatibility)
    if controller_state is None:
        controller_state = {'risk_ema': 0.0, 'internal_zone': 'safe', 'episode_id': -1}
    
    # Get risk score and update EMA
    risk_score = 0.0
    if use_state and edon_state_raw is not None:
        risk_score = compute_risk_score(edon_state_raw)
        alpha = 0.15
        controller_state['risk_ema'] = alpha * risk_score + (1 - alpha) * controller_state.get('risk_ema', 0.0)
    else:
        controller_state['risk_ema'] = 0.0
    
    risk_ema = controller_state['risk_ema']
    
    # Determine internal zone (from +7.0% config)
    internal_zone = controller_state.get('internal_zone', tilt_zone)
    if tilt_zone == "safe" and risk_ema >= 0.5 and max_tilt > 0.6 * config.SAFE_LIMIT:
        internal_zone = "prefall"
    elif tilt_zone in ("prefall", "fail"):
        internal_zone = tilt_zone
    elif internal_zone == "prefall":
        if max_tilt < config.SAFE_LIMIT * 0.6 and risk_ema < 0.3:
            internal_zone = "safe"
    else:
        internal_zone = "safe"
    
    controller_state['internal_zone'] = internal_zone
    
    # ============================
    # ADAPTIVE INSTABILITY SCORING & PHASE MANAGEMENT
    # ============================
    
    # Compute instability score using adaptive controller
    instability_score = controller.compute_instability_score(
        obs, edon_state_raw, tilt_zone, risk_ema
    )
    
    # Update smoothed signals
    controller.update_smoothed_signals(tilt_magnitude, vel_norm, instability_score)
    
    # Update phase with hysteresis
    phase = controller.update_phase(controller.adaptive_state.instability_score)
    
    # Track for logging
    controller.adaptive_state.instability_history.append(instability_score)
    controller.adaptive_state.step_count += 1
    
    # ============================
    # COMPUTE BASE PD CORRECTIONS
    # ============================
    
    Kp_roll = EDON_BASE_KP_ROLL
    Kp_pitch = EDON_BASE_KP_PITCH
    Kd_roll = EDON_BASE_KD_ROLL
    Kd_pitch = EDON_BASE_KD_PITCH
    
    # Compute corrections with proper sign (oppose tilt)
    corr_roll = -Kp_roll * roll - Kd_roll * roll_velocity
    corr_pitch = -Kp_pitch * pitch - Kd_pitch * pitch_velocity
    
    # Ensure corrections oppose tilt direction
    if abs(roll) > 0.01:
        if (roll > 0 and corr_roll > 0) or (roll < 0 and corr_roll < 0):
            corr_roll *= -1.0
    if abs(pitch) > 0.01:
        if (pitch > 0 and corr_pitch > 0) or (pitch < 0 and corr_pitch < 0):
            corr_pitch *= -1.0
    
    correction = map_torso_correction_to_action(corr_roll, corr_pitch, baseline_action)
    
    # Verify direction is stabilizing
    is_tilt_stabilizing = True
    if len(correction) >= 2:
        if abs(roll) > 0.01:
            roll_opposing = (roll > 0 and correction[0] < 0) or (roll < 0 and correction[0] > 0)
            if not roll_opposing:
                correction[0] *= -1.0
                is_tilt_stabilizing = False
        if abs(pitch) > 0.01:
            pitch_opposing = (pitch > 0 and correction[1] < 0) or (pitch < 0 and correction[1] > 0)
            if not pitch_opposing:
                correction[1] *= -1.0
                is_tilt_stabilizing = False
        
        if len(correction) > 4:
            correction[4:] = 0.0
    
    # ============================
    # ADAPTIVE GAIN COMPUTATION
    # ============================
    
    # Compute adaptive gain based on phase
    adaptive_gain = controller.compute_adaptive_gain()
    controller.adaptive_state.gain_history.append(adaptive_gain)
    
    # Extract EDON state variables for legacy compatibility
    p_chaos = 0.0
    p_stress = 0.0
    fall_risk = risk_ema
    catastrophic_risk = 1.0 if internal_zone == "fail" else (risk_ema * 1.2)
    catastrophic_risk = max(0.0, min(1.0, catastrophic_risk))
    
    if use_state and edon_state_raw is not None:
        p_chaos = edon_state_raw.get("p_chaos", 0.0)
        p_stress = edon_state_raw.get("p_stress", 0.0)
        if not isinstance(p_chaos, (int, float)):
            p_chaos = 0.0
        if not isinstance(p_stress, (int, float)):
            p_stress = 0.0
        p_chaos = max(0.0, min(1.0, p_chaos))
        p_stress = max(0.0, min(1.0, p_stress))
    
    # ============================
    # PHASE-BASED TORQUE BLENDING
    # ============================
    
    # Compute prefall torque (from correction)
    prefall_torque = correction.copy()
    
    PREFALL_MIN = 0.18
    PREFALL_MAX = 0.65
    prefall_gain = PREFALL_MIN + (PREFALL_MAX - PREFALL_MIN) * fall_risk
    prefall_gain = min(PREFALL_MAX, max(PREFALL_MIN, prefall_gain))
    
    if internal_zone in ("prefall", "fail"):
        prefall_torque *= prefall_gain
    else:
        prefall_torque *= 0.0
    
    # Compute safe torque
    safe_torque = np.zeros_like(baseline_action)
    if catastrophic_risk > 0.75:
        SAFE_GAIN = 0.12
        safe_torque[0] = -0.15 * roll * SAFE_GAIN
        safe_torque[1] = -0.15 * pitch * SAFE_GAIN
        if len(safe_torque) > 4:
            safe_torque[4:] = 0.0
    else:
        safe_torque *= 0.0
    
    safe_active = catastrophic_risk > 0.75
    if controller_state is not None:
        controller_state["safe_active"] = safe_active
    
    # Get phase-based blending weights
    w_prefall, w_safe = controller.get_torque_weights()
    
    # Blend torques based on phase
    combined_torque = w_prefall * prefall_torque + w_safe * safe_torque
    
    # ============================
    # DIAGNOSTIC TRACKING (torque activity)
    # ============================
    
    # Track prefall torque activity
    prefall_mag = np.linalg.norm(prefall_torque)
    prefall_is_active = prefall_mag > 1e-6
    if prefall_is_active:
        controller.adaptive_state.prefall_magnitudes.append(prefall_mag)
        controller.adaptive_state.prefall_active_count += 1
    
    # Track safe torque activity
    safe_mag = np.linalg.norm(safe_torque)
    safe_is_active = safe_mag > 1e-6
    if safe_is_active:
        controller.adaptive_state.safe_magnitudes.append(safe_mag)
        controller.adaptive_state.safe_active_count += 1
    
    # ============================
    # FINAL ACTION BLEND
    # ============================
    
    # Apply adaptive gain to combined torque
    # Note: adaptive_gain already includes base_edon_gain scaling
    edon_contribution = adaptive_gain * combined_torque
    final_action = baseline_action + edon_contribution
    
    # Debug: Log gain usage on first call
    if controller_state is not None and not controller_state.get("_logged_gain_usage", False):
        print(f"[EDON-ADAPTIVE] base_gain={edon_gain:.3f} adaptive_gain={adaptive_gain:.3f} phase={phase}")
        controller_state["_logged_gain_usage"] = True
    
    # Get phase-dependent clamp ratio
    clamp_ratio = controller.get_clamp_ratio()
    
    # Clamp to prevent overshoot (phase-dependent)
    final_action = clamp_action_relative_to_baseline(
        baseline_action,
        final_action,
        max_ratio=clamp_ratio
    )
    
    # Safety clamp (phase-dependent)
    action = clamp_action_relative_to_baseline(
        baseline_action=baseline_action,
        edon_action=final_action,
        max_ratio=clamp_ratio
    )
    
    # Final clip
    action = np.clip(action, -1.0, 1.0)
    
    # Track action delta (diagnostic)
    delta = action - baseline_action
    delta_mag = np.linalg.norm(delta)
    controller.adaptive_state.delta_magnitudes.append(delta_mag)
    
    # Build debug info
    baseline_norm = np.linalg.norm(baseline_action)
    debug_info = {
        'tilt_zone': tilt_zone,
        'internal_zone': internal_zone,
        'risk_score': risk_score,
        'risk_ema': risk_ema,
        'is_tilt_stabilizing': is_tilt_stabilizing,
        'adaptive_gain': adaptive_gain,
        'phase': phase,
        'instability_score': instability_score,
        'prefall_gain': prefall_gain if internal_zone in ("prefall", "fail") else 0.0,
        'safe_active': safe_active,
        'w_prefall': w_prefall,
        'w_safe': w_safe,
        'corr_ratio': np.linalg.norm(combined_torque) / baseline_norm if baseline_norm > 1e-6 else 0.0
    }
    
    return action, debug_info


def edon_controller(
    obs: dict,
    edon_state: Optional[dict] = None,
    edon_gain: float = 0.5,
    edon_arch: str = "v5_heuristic",
    debug: bool = False
) -> np.ndarray:
    """
    Control policy with EDON integration (V5 Architecture).
    
    Uses modular EDON v5 architecture with pluggable policies.
    Default policy (v5_heuristic) maintains v4.2 behavior.
    
    CRITICAL: For learned policies (v6_learned), only uses EDON in prefall/warning/recovery zones.
    Safe zones always use baseline to prevent the model from acting when it shouldn't.
    
    Args:
        obs: Observation dictionary
        edon_state: EDON state dictionary (from EDON API)
        edon_gain: User-provided gain parameter (scales entire EDON contribution)
        edon_arch: EDON architecture selection (default: "v5_heuristic")
        debug: Debug flag
    
    Returns:
        Action array with EDON corrections applied (or baseline if in safe zone for learned policy)
    """
    # Debug: Print gain on first call to verify it's being used
    if not hasattr(edon_controller, "_printed_gain"):
        print(f"[EDON-CONTROLLER] gain={edon_gain} arch={edon_arch} (this scales the entire EDON contribution)")
        edon_controller._printed_gain = True
    
    if edon_state is None:
        return baseline_controller(obs, None)
    
    # Get core instance (needed for both routing and policy execution)
    core = get_edon_core(base_gain=edon_gain, edon_arch=edon_arch)
    
    # For learned policies, check zone/phase to decide if EDON should be used
    if edon_arch in ("v6_learned", "v6_1_learned"):
        # Extract features to determine zone
        features = core.extract_features(obs, edon_state, controller_state=None)
        tilt_zone = features.tilt_zone
        
        # Determine if learned policy should be used
        use_learned_policy = False
        
        # Routing logic:
        # - Learned policy used when: tilt_zone in ("prefall", "fail") OR (tilt_zone == "safe" AND phase in ("warning", "recovery"))
        # - Baseline used when: tilt_zone == "safe" AND phase == "stable"
        if tilt_zone in ("prefall", "fail"):
            # Prefall/fail: always use learned policy
            use_learned_policy = True
        else:
            # Safe zone: need to check phase
            # Update state to get current phase (this computes instability and phase)
            core.update_state(features)
            phase = core.state.phase
            
            # Route based on phase
            if phase in ("warning", "recovery"):
                # Warning/recovery phase: use learned policy
                use_learned_policy = True
            else:
                # Safe zone + stable phase: use baseline only
                use_learned_policy = False
        
        # Track usage for per-episode logging
        if not hasattr(core, "_episode_policy_steps"):
            core._episode_policy_steps = 0
            core._episode_baseline_steps = 0
        
        if use_learned_policy:
            core._episode_policy_steps += 1
            # Log once per episode when policy becomes active
            if core._episode_policy_steps == 1:
                phase = core.state.phase if tilt_zone == "safe" else "N/A"
                print(f"[EDON] Policy ACTIVE in {tilt_zone} zone (phase={phase}) - using learned policy")
        else:
            core._episode_baseline_steps += 1
            # Log once per episode when baseline is used
            if core._episode_baseline_steps == 1:
                phase = core.state.phase
                print(f"[EDON] Policy BYPASSED in {tilt_zone} zone (phase={phase}) - using baseline")
            return baseline_controller(obs, None)
    
    # For heuristic policies or when learned policy should be active, use EDON
    baseline_action = np.array(baseline_controller(obs, None))
    
    # Use EDON v5 core to compute action
    # Note: For v6_learned, we've already extracted features and updated state if needed
    # compute_action will do it again, but that's okay - it ensures consistency
    final_action = core.compute_action(
        baseline_action=baseline_action,
        obs=obs,
        edon_state_raw=edon_state,
        controller_state=None
    )
    
    return final_action


# ============================================================================
# Environment Setup
# ============================================================================

def make_humanoid_env(seed: Optional[int] = None, profile: Optional[str] = None):
    """Create humanoid environment with optional stress profile."""
    env = MockHumanoidEnv(seed=seed)
    if profile:
        # Store profile in env for potential use
        env.stress_profile = profile
    return env


# ============================================================================
# Main Evaluation Loop
# ============================================================================

def main():
    """Main evaluation runner."""
    parser = argparse.ArgumentParser(description="EDON Humanoid Evaluation")
    parser.add_argument("--mode", choices=["baseline", "edon"], required=True,
                       help="Evaluation mode")
    parser.add_argument("--episodes", type=int, default=30,
                       help="Number of episodes to run")
    parser.add_argument("--profile", choices=list(STRESS_PROFILES.keys()), default="medium_stress",
                       help="Stress profile")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file path")
    parser.add_argument("--edon-gain", type=float, default=0.75,
                       help="EDON gain (0.0-1.0)")
    parser.add_argument("--edon-url", type=str, default="http://127.0.0.1:8001",
                       help="EDON server URL")
    parser.add_argument("--edon-arch", type=str, default="v5_heuristic",
                       choices=["v5_heuristic", "v6_learned", "v6_1_learned", "v7_learned", "v8_strategy"],
                       help="EDON core architecture/policy to use")
    parser.add_argument("--edon-log-train", action="store_true",
                       help="If set, log per-step EDON data for training (JSONL)")
    parser.add_argument("--edon-log-path", type=str, default=None,
                       help="Optional path for EDON training log JSONL file. If not set and --edon-log-train is true, use logs/edon_train_<profile>_<timestamp>.jsonl")
    parser.add_argument("--randomize-env", action="store_true",
                       help="Enable environment randomization")
    parser.add_argument("--edon-score", action="store_true",
                       help="Compute and print EDON episode score")
    
    args = parser.parse_args()
    
    # Debug: Print received arguments (VERIFY these change when you switch modes/profiles/gain)
    print("="*70)
    print("[RUN-EVAL] ARGUMENT PARSING")
    print("="*70)
    print(f"[RUN-EVAL] mode={args.mode} (MUST be 'baseline' or 'edon')")
    print(f"[RUN-EVAL] profile={args.profile} (MUST match requested profile)")
    print(f"[RUN-EVAL] edon_gain={args.edon_gain if args.mode == 'edon' else None} (only used in EDON mode)")
    print(f"[RUN-EVAL] episodes={args.episodes} seed={args.seed}")
    print("="*70)
    
    # Set config
    config.STRESS_PROFILE = args.profile
    
    # Create base environment with profile
    base_env = make_humanoid_env(seed=args.seed, profile=args.profile)
    print(f"[RUN-EVAL] using_env_profile={args.profile} (VERIFY this matches --profile argument)")
    if hasattr(base_env, 'stress_profile'):
        print(f"[RUN-EVAL] env.stress_profile={base_env.stress_profile}")
    
    # Check if v8_strategy architecture - wrap environment with v8
    v8_strategy_policy = None
    v8_fail_risk_model = None
    if args.mode == "edon" and args.edon_arch == "v8_strategy":
        print(f"[RUN-EVAL] Initializing v8_strategy architecture...")
        try:
            import torch
            from env.edon_humanoid_env_v8 import EdonHumanoidEnvV8
            from training.edon_v8_policy import EdonV8StrategyPolicy, pack_observation_v8
            from training.fail_risk_model import FailRiskModel
            
            # Load fail-risk model (try newest fixed version first, then fallback)
            fail_risk_model_path = Path("models/edon_fail_risk_v1_fixed_v2.pt")
            if not fail_risk_model_path.exists():
                fail_risk_model_path = Path("models/edon_fail_risk_v1_fixed.pt")
            if not fail_risk_model_path.exists():
                fail_risk_model_path = Path("models/edon_fail_risk_v1.pt")
            if fail_risk_model_path.exists():
                checkpoint = torch.load(fail_risk_model_path, map_location="cpu", weights_only=False)
                input_size = checkpoint.get("input_size", 15)
                v8_fail_risk_model = FailRiskModel(input_size=input_size)
                v8_fail_risk_model.load_state_dict(checkpoint["model_state_dict"])
                v8_fail_risk_model.eval()
                print(f"[RUN-EVAL] Loaded fail-risk model from {fail_risk_model_path}")
            else:
                print(f"[RUN-EVAL] Warning: Fail-risk model not found at {fail_risk_model_path}, using fail-risk=0.0")
            
            # Load v8 strategy policy
            # Try memory_features model first (newest), then intervention_first, then fixed, then others
            strategy_model_path = Path("models/edon_v8_strategy_memory_features.pt")
            if not strategy_model_path.exists():
                strategy_model_path = Path("models/edon_v8_strategy_intervention_first.pt")
            if not strategy_model_path.exists():
                strategy_model_path = Path("models/edon_v8_strategy_v1_fixed.pt")
            if not strategy_model_path.exists():
                strategy_model_path = Path("models/edon_v8_strategy_v1_no_reflex.pt")
            if not strategy_model_path.exists():
                strategy_model_path = Path("models/edon_v8_strategy_v1.pt")
            if strategy_model_path.exists():
                checkpoint = torch.load(strategy_model_path, map_location="cpu", weights_only=False)
                input_size = checkpoint.get("input_size")
                
                # Determine input size if not in checkpoint (with stacked observations)
                if input_size is None:
                    from training.edon_v8_policy import pack_stacked_observation_v8
                    test_obs = base_env.reset()
                    test_baseline = baseline_controller(test_obs, edon_state=None)
                    test_input = pack_stacked_observation_v8(
                        obs=test_obs,
                        baseline_action=np.array(test_baseline),
                        fail_risk=0.0,
                        instability_score=0.0,
                        phase="stable",
                        obs_history=None,
                        near_fail_history=None,
                        obs_vec_history=None,
                        stack_size=8
                    )
                    input_size = len(test_input)
                    print(f"[RUN-EVAL] Computed input size with stacked obs: {input_size}")
                
                v8_strategy_policy = EdonV8StrategyPolicy(input_size=input_size)
                v8_strategy_policy.load_state_dict(checkpoint["policy_state_dict"])
                v8_strategy_policy.eval()
                print(f"[RUN-EVAL] Loaded v8 strategy policy from {strategy_model_path}")
            else:
                raise FileNotFoundError(f"V8 strategy model not found at {strategy_model_path}")
            
            # Wrap environment with v8
            device = "cuda" if torch.cuda.is_available() else "cpu"
            env = EdonHumanoidEnvV8(
                strategy_policy=v8_strategy_policy,
                fail_risk_model=v8_fail_risk_model,
                base_env=base_env,
                seed=args.seed,
                profile=args.profile,
                device=device
            )
            print(f"[RUN-EVAL] Environment wrapped with EdonHumanoidEnvV8")
        except Exception as e:
            print(f"[RUN-EVAL] ERROR: Failed to initialize v8_strategy: {e}")
            import traceback
            traceback.print_exc()
            print(f"[RUN-EVAL] Falling back to base environment")
            env = base_env
    else:
        env = base_env
    
    # Setup randomizer - ALWAYS use profile (not just when --randomize-env is set)
    # The profile determines the difficulty level (push forces, noise, etc.)
    stress_profile = get_stress_profile(args.profile)
    randomizer = EnvironmentRandomizer(stress_profile=stress_profile, seed=args.seed)
    print(f"[RUN-EVAL] randomizer_profile={args.profile} (push_prob={stress_profile.push_probability:.2f}, noise_std={stress_profile.sensor_noise_std:.3f})")
    
    # Setup controller
    use_edon = (args.mode == "edon")
    edon_client = None
    
    if use_edon:
        if EdonClient is None:
            print("ERROR: EDON SDK not available. Cannot run EDON mode.")
            sys.exit(1)
        
        try:
            edon_client = EdonClient(base_url=args.edon_url)
            # Check health
            try:
                import requests
                health_url = f"{args.edon_url}/health"
                resp = requests.get(health_url, timeout=2.0)
                if resp.status_code == 200:
                    print(f"[OK] EDON server is running")
                else:
                    print(f"[WARNING] EDON server health check returned {resp.status_code}")
            except Exception as e:
                print(f"[WARNING] Could not check EDON server health: {e}")
                print(f"[WARNING] Continuing anyway...")
        except Exception as e:
            print(f"[WARNING] Failed to initialize EDON client: {e}")
            print(f"[WARNING] Continuing without EDON...")
            use_edon = False
    
    # Create controller function - MUST be different for baseline vs EDON
    # For v8_strategy, controller is bypassed (v8 env handles control internally)
    if use_edon and args.edon_arch == "v8_strategy":
        # v8 environment handles control internally, controller is no-op
        # Use baseline_controller function directly (defined at module level)
        def controller(obs, edon_state=None):
            # Return baseline action (v8 env will ignore it and compute internally)
            return baseline_controller(obs, edon_state)
        controller_name = "v8_internal"
        controller_gain = args.edon_gain
        controller_type = f"EDONController-v8(strategy+reflex)"
    elif use_edon:
        def controller(obs, edon_state=None):
            return edon_controller(obs, edon_state, edon_gain=args.edon_gain, edon_arch=args.edon_arch)
        controller_name = "edon_controller"
        controller_gain = args.edon_gain
        controller_type = f"EDONController-v5({args.edon_arch})"
    else:
        controller = baseline_controller
        controller_name = "baseline_controller"
        controller_gain = None
        controller_type = "BaselineController"
    
    print(f"[RUN-EVAL] controller={controller_type} ({controller_name}) gain={controller_gain}")
    print(f"[RUN-EVAL] use_edon={use_edon} (MUST be True for EDON mode, False for baseline)")
    
    # Create runner
    runner = HumanoidRunner(
        env=env,
        controller=controller,
        edon_client=edon_client,
        randomizer=randomizer,
        use_edon=use_edon
    )
    
    # Run episodes
    print(f"\n{'='*70}")
    print(f"Running {args.mode.upper()} evaluation")
    print(f"  Episodes: {args.episodes}")
    print(f"  Profile: {args.profile}")
    print(f"  Seed: {args.seed}")
    if use_edon:
        print(f"  EDON gain: {args.edon_gain}")
    print(f"{'='*70}\n")
    
    episode_metrics_list = []
    controller_state = {
        'risk_ema': 0.0, 
        'internal_zone': 'safe', 
        'episode_id': -1,
        'step_idx': 0,
        'debug_edon': True  # Enable debug logging
    }
    
    # Initialize EDON v5 core if in EDON mode
    train_logger = None
    if use_edon:
        # Setup training logger if requested
        if args.edon_log_train:
            if args.edon_log_path:
                log_path = args.edon_log_path
            else:
                # Default path: logs/edon_train_<profile>_<timestamp>.jsonl
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_dir = os.path.join("logs")
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, f"edon_train_{args.profile}_{ts}.jsonl")
            
            run_metadata = {
                "profile": args.profile,
                "episodes": args.episodes,
                "seed": args.seed,
                "mode": args.mode,
                "edon_gain": args.edon_gain,
                "edon_arch": args.edon_arch,
            }
            train_logger = EdonTrainLogger(log_path, run_metadata)
            print(f"[EDON-TRAIN-LOG] Logging to: {log_path}")
        
        # Skip EDON core creation for v8_strategy (v8 uses environment wrapper)
        if args.edon_arch != "v8_strategy":
            global _edon_core_instance
            _edon_core_instance = get_edon_core(
                base_gain=args.edon_gain,
                edon_arch=args.edon_arch,
                train_logger=train_logger,
                profile=args.profile,
                mode=args.mode
            )
    
    for episode_id in range(args.episodes):
        # Reset EDON core state for new episode (skip for v8_strategy - handled by env)
        if use_edon and args.edon_arch != "v8_strategy" and _edon_core_instance is not None:
            _edon_core_instance.reset_episode(episode_id)
            # Reset per-episode counters
            _edon_core_instance._episode_policy_steps = 0
            _edon_core_instance._episode_baseline_steps = 0
        
        print(f"Episode {episode_id + 1}/{args.episodes}...", end=" ", flush=True)
        metrics = runner.run_episode(episode_id=episode_id)
        episode_metrics_list.append(metrics)
        
        # Log adaptive behavior summary (v5) - skip for v8_strategy
        if use_edon and args.edon_arch != "v8_strategy" and _edon_core_instance is not None:
            _edon_core_instance.log_episode_summary(
                episode_id,
                interventions=metrics.interventions,
                stability_score=metrics.stability_score
            )
        
        # For v8, extract v8-specific info from episode if available
        if use_edon and args.edon_arch == "v8_strategy":
            # v8-specific info is in env.step() info dict, but we need to collect it during episode
            # For now, just print standard metrics
            print(f"Interventions: {metrics.interventions}, Stability: {metrics.stability_score:.4f}")
        else:
            print(f"Interventions: {metrics.interventions}, Stability: {metrics.stability_score:.4f}")
    
    # Close training logger if it exists
    if use_edon and _edon_core_instance is not None and _edon_core_instance.train_logger is not None:
        _edon_core_instance.train_logger.close()
        print(f"[EDON-TRAIN-LOG] Training log closed")
    
    # Aggregate results
    run_metrics = aggregate_run_metrics(episode_metrics_list, mode=args.mode)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "mode": args.mode,
        "profile": args.profile,
        "episodes": args.episodes,
        "seed": args.seed,
        "edon_gain": args.edon_gain if use_edon else None,
        "run_metrics": {
            "interventions_per_episode": run_metrics.interventions_per_episode,
            "freeze_events_per_episode": run_metrics.freeze_events_per_episode,
            "stability_avg": run_metrics.stability_avg,
            "avg_episode_length": run_metrics.avg_episode_length,
            "success_rate": run_metrics.success_rate,
        },
        "episodes": [m.to_dict() for m in episode_metrics_list]
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Evaluation Complete")
    print(f"{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Interventions/episode: {run_metrics.interventions_per_episode:.2f}")
    print(f"  Freeze events/episode: {run_metrics.freeze_events_per_episode:.2f}")
    print(f"  Stability (avg): {run_metrics.stability_avg:.4f}")
    print(f"  Episode length (avg): {run_metrics.avg_episode_length:.1f} steps")
    print(f"  Success rate: {run_metrics.success_rate * 100:.1f}%")
    
    # Optionally compute and print EDON episode score
    if args.edon_score:
        episode_summary = {
            "interventions_per_episode": run_metrics.interventions_per_episode,
            "stability_avg": run_metrics.stability_avg,
            "avg_episode_length": run_metrics.avg_episode_length
        }
        
        # Use v8 score if v8_strategy, otherwise use standard score
        if args.mode == "edon" and args.edon_arch == "v8_strategy":
            from metrics.edon_v8_metrics import compute_episode_score_v8, compute_episode_metrics_v8
            
            # Compute v8-specific metrics from episode data
            v8_episode_data = []
            for ep_metrics in episode_metrics_list:
                # Convert EpisodeMetrics to step records format (simplified)
                # In practice, you'd want to store step-level data during episodes
                ep_dict = {
                    "interventions": ep_metrics.interventions,
                    "stability_score": ep_metrics.stability_score,
                    "episode_length": ep_metrics.episode_length
                }
                v8_episode_data.append(ep_dict)
            
            # Add v8-specific fields if available from episode metadata
            if len(episode_metrics_list) > 0 and hasattr(episode_metrics_list[0], 'metadata'):
                # Try to extract v8 metrics from metadata
                for i, ep_metrics in enumerate(episode_metrics_list):
                    if "time_to_first_intervention" in ep_metrics.metadata:
                        episode_summary["time_to_first_intervention"] = ep_metrics.metadata["time_to_first_intervention"]
                    if "near_fail_density" in ep_metrics.metadata:
                        episode_summary["near_fail_density"] = ep_metrics.metadata["near_fail_density"]
            
            edon_score = compute_episode_score_v8(episode_summary)
            print(f"\nEDON v8 Episode Score: {edon_score:.2f}")
            print(f"  (Components: interventions={episode_summary['interventions_per_episode']:.2f}, "
                  f"stability={episode_summary['stability_avg']:.4f}, "
                  f"length={episode_summary['avg_episode_length']:.1f})")
            if "time_to_first_intervention" in episode_summary:
                print(f"  time_to_intervention={episode_summary.get('time_to_first_intervention', 'N/A')}")
            if "near_fail_density" in episode_summary:
                print(f"  near_fail_density={episode_summary.get('near_fail_density', 0.0):.4f}")
        else:
            edon_score = compute_episode_score(episode_summary)
            print(f"\nEDON Episode Score: {edon_score:.2f}")
            print(f"  (Components: interventions={episode_summary['interventions_per_episode']:.2f}, "
                  f"stability={episode_summary['stability_avg']:.4f}, "
                  f"length={episode_summary['avg_episode_length']:.1f})")


if __name__ == "__main__":
    main()
