"""
EDON Episode Score and Reward Functions

Defines the formal EDON scoring system for evaluating episode performance
and computing per-step rewards for reinforcement learning.

The EDON score rewards:
- Fewer interventions (safety events)
- Lower instability (better stability scores)
- Longer episodes (more successful operation)
"""

from typing import Dict, Any, Optional
import numpy as np


def compute_episode_score(summary: Dict[str, Any]) -> float:
    """
    Compute EDON episode score from episode summary.
    
    Higher score = better performance.
    
    Components:
    - Fewer interventions = higher score
    - Lower instability (stability_avg) = higher score
    - Longer episodes = higher score (up to a point)
    
    Args:
        summary: Episode summary dict with keys:
            - interventions_per_episode (or interventions): float
            - stability_avg (or stability_score): float
            - avg_episode_length (or episode_length): float (optional)
    
    Returns:
        EDON episode score (float, typically in range [0, 100+])
    """
    # Extract metrics with fallback keys
    interventions = summary.get("interventions_per_episode") or summary.get("interventions", 0)
    stability = summary.get("stability_avg") or summary.get("stability_score", 0.0)
    episode_length = summary.get("avg_episode_length") or summary.get("episode_length", 0)
    
    # Normalize interventions (fewer is better)
    # Typical range: 0-50 interventions per episode
    # Score: 100 - (interventions * 2), clamped to [0, 100]
    intervention_score = max(0.0, 100.0 - (interventions * 2.0))
    
    # Normalize stability (lower is better)
    # Typical range: 0.0-0.1 (lower variance = more stable)
    # Score: 100 * (1.0 - min(1.0, stability * 10)), clamped to [0, 100]
    # If stability is 0.02, score = 100 * (1.0 - 0.2) = 80
    stability_score = 100.0 * (1.0 - min(1.0, stability * 10.0))
    
    # Episode length bonus (longer episodes = more successful)
    # Typical range: 0-1000 steps
    # Bonus: min(20, episode_length / 50), capped at 20 points
    length_bonus = min(20.0, episode_length / 50.0)
    
    # Weighted combination
    # Interventions and stability are primary (40% each)
    # Length is secondary (20%)
    total_score = (
        0.4 * intervention_score +
        0.4 * stability_score +
        0.2 * length_bonus
    )
    
    return float(total_score)


def step_reward(
    prev_state: Optional[Dict[str, Any]],
    next_state: Dict[str, Any],
    info: Optional[Dict[str, Any]] = None,
    w_intervention: float = 20.0,
    w_stability: float = 1.0,
    w_torque: float = 0.1,
    return_components: bool = False
) -> float:
    """
    Compute per-step reward for RL training.
    
    INTERVENTION-FIRST REWARD: Strongly prioritizes reducing interventions.
    
    Objectives (in priority order):
    1. Strongly penalize interventions (primary goal) - weight: w_intervention
    2. Penalize instability (tilt / velocity) - weight: w_stability
    3. Penalize large action deltas (smoothness) - weight: w_torque
    4. Small alive bonus
    
    Args:
        prev_state: Previous observation/state dict (optional)
        next_state: Current observation/state dict
        info: Step info dict (may contain intervention flags, edon_delta)
        w_intervention: Weight for intervention penalty (default: 20.0)
        w_stability: Weight for stability penalty (default: 1.0)
        w_torque: Weight for torque/action penalty (default: 0.1)
    
    Returns:
        Step reward (float)
    """
    # Extract stability metrics
    roll = abs(next_state.get("roll", 0.0))
    pitch = abs(next_state.get("pitch", 0.0))
    roll_velocity = abs(next_state.get("roll_velocity", 0.0))
    pitch_velocity = abs(next_state.get("pitch_velocity", 0.0))
    
    # Compute magnitudes once
    tilt_mag = roll + pitch
    vel_mag = roll_velocity + pitch_velocity
    
    # Initialize reward
    reward = 0.0
    
    # CRITICAL: Strongly penalize interventions (PRIMARY GOAL)
    # Use configurable weight to make intervention avoidance the top priority
    intervention = 1.0 if (info and (info.get("intervention", False) or info.get("fallen", False))) else 0.0
    reward -= w_intervention * intervention  # Strong penalty per intervention event
    
    # Penalize instability (tilt) - but with smaller weight than interventions
    # SCALED DOWN: Reduced by ~10x so stability doesn't dominate reward
    # tilt_mag is typically in [0, ~0.7] for normal operation
    # Target: ~-0.3 per step instead of ~-3.0 per step
    tilt_penalty = w_stability * 0.8 * tilt_mag  # Reduced from 8.0 to 0.8 (10x reduction)
    if tilt_mag > 0.2:  # Extra penalty for dangerous tilt
        tilt_penalty += w_stability * 0.5 * (tilt_mag - 0.2)  # Reduced from 5.0 to 0.5
    reward -= tilt_penalty
    
    # Penalize instability (velocity) - also affects stability
    # SCALED DOWN: Reduced by ~10x
    # vel_mag is typically in [0, ~2.0] for normal operation
    reward -= w_stability * 0.3 * vel_mag  # Reduced from 3.0 to 0.3 (10x reduction)
    
    # CRITICAL: Positive reward for staying alive (matches length_bonus in EDON Score)
    # EDON Score: length_bonus = min(20, episode_length / 50)
    # For 300-step episode: length_bonus = 6.0 points (out of 100 total)
    # For 400-step episode: length_bonus = 8.0 points
    # We want reward to be POSITIVELY correlated with length
    # Strategy: Make alive bonus large enough to offset penalties for longer episodes
    # Increased to 0.8 to better incentivize longer episodes
    alive = 1.0 if not (info and info.get("fallen", False)) else 0.0
    reward += 0.8 * alive  # Increased alive bonus to better match length_bonus
    
    # Penalize large action deltas (smoothness, prevents overcorrection)
    # Extract edon_delta from info if available
    edon_delta = None
    if info:
        edon_delta = info.get("edon_delta", None)
    
    if edon_delta is not None:
        # Compute L2 magnitude of action delta
        if isinstance(edon_delta, np.ndarray):
            action_penalty = float(np.linalg.norm(edon_delta))
        elif isinstance(edon_delta, (list, tuple)):
            action_penalty = float(np.linalg.norm(np.array(edon_delta)))
        else:
            action_penalty = float(abs(edon_delta))
        
        # Penalize large deltas (but don't explode - scaled for smoothness)
        reward -= w_torque * action_penalty
    
    # HIDDEN INSTABILITY PENALTIES (prevents "fake calm, real chaos")
    # These detect instability that might not show up in simple tilt/velocity metrics
    
    # 1. High-frequency oscillation penalty
    # Track recent state history to detect oscillations
    if prev_state is not None:
        prev_roll = abs(prev_state.get("roll", 0.0))
        prev_pitch = abs(prev_state.get("pitch", 0.0))
        prev_roll_vel = abs(prev_state.get("roll_velocity", 0.0))
        prev_pitch_vel = abs(prev_state.get("pitch_velocity", 0.0))
        
        # Detect rapid oscillations (sign changes in velocity)
        roll_oscillation = 1.0 if (prev_roll_vel * roll_velocity < 0) else 0.0
        pitch_oscillation = 1.0 if (prev_pitch_vel * pitch_velocity < 0) else 0.0
        
        # High-frequency penalty (oscillating rapidly)
        # SCALED DOWN: Reduced by ~10x
        oscillation_penalty = w_stability * 0.2 * (roll_oscillation + pitch_oscillation)  # Reduced from 2.0 to 0.2
        reward -= oscillation_penalty
    
    # 2. Phase-lag penalty (velocity and position out of phase = instability)
    # SCALED DOWN: Reduced by ~10x
    # When tilt is high but velocity is low (or vice versa), there's phase lag
    tilt_vel_product = tilt_mag * vel_mag
    if tilt_mag > 0.1 and vel_mag < 0.5:  # High tilt, low velocity = phase lag
        phase_lag_penalty = w_stability * 0.3 * tilt_mag  # Reduced from 3.0 to 0.3
        reward -= phase_lag_penalty
    elif vel_mag > 1.0 and tilt_mag < 0.1:  # High velocity, low tilt = also phase lag
        phase_lag_penalty = w_stability * 0.2 * vel_mag  # Reduced from 2.0 to 0.2
        reward -= phase_lag_penalty
    
    # 3. Control jerk penalty (Δu/Δt) - rapid action changes
    if info and edon_delta is not None:
        # Compute action change rate (jerk)
        if isinstance(edon_delta, np.ndarray):
            jerk_mag = float(np.linalg.norm(edon_delta))
        elif isinstance(edon_delta, (list, tuple)):
            jerk_mag = float(np.linalg.norm(np.array(edon_delta)))
        else:
            jerk_mag = float(abs(edon_delta))
        
        # Penalize high jerk (rapid control changes)
        if jerk_mag > 0.5:  # Threshold for "high jerk"
            jerk_penalty = w_torque * 1.5 * (jerk_mag - 0.5)
            reward -= jerk_penalty
    
    # Clamp to reasonable range
    # Target: 300-400 step episodes yield -200 to -800 total reward
    # With alive bonus of 0.4, 300 steps = +120, so net should be -200 to -800
    # This means per-step net should be roughly -0.67 to -2.67
    reward = max(-3.0, min(0.5, reward))
    
    if return_components:
        # Return breakdown for diagnostics
        return {
            "total": float(reward),
            "intervention": -w_intervention * intervention,
            "stability_tilt": -tilt_penalty,
            "stability_velocity": -w_stability * 3.0 * vel_mag,
            "stability_oscillation": -oscillation_penalty if prev_state is not None else 0.0,
            "stability_phase_lag": -(phase_lag_penalty if (tilt_mag > 0.1 and vel_mag < 0.5) or (vel_mag > 1.0 and tilt_mag < 0.1) else 0.0),
            "torque_action": -w_torque * action_penalty if edon_delta is not None else 0.0,
            "torque_jerk": -(jerk_penalty if (info and edon_delta is not None and jerk_mag > 0.5) else 0.0),
            "alive_bonus": 0.8 * alive
        }
    
    return float(reward)

