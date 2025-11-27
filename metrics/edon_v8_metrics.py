"""
EDON v8 Metrics

Metrics specific to v8 architecture that include:
- Time to first intervention
- Average fail-risk
- Near-fail steps (fail_risk > 0.6)
- Near-fail density
- Recovery success rate
- EDON Score v8
"""

from typing import Dict, Any, List, Optional
import numpy as np


def compute_time_to_first_intervention(episode_data: List[Dict[str, Any]]) -> Optional[float]:
    """
    Compute time (in steps) to first intervention.
    
    Args:
        episode_data: List of step records with 'info' dicts containing 'intervention' flag
    
    Returns:
        Step index of first intervention, or None if no intervention
    """
    for i, step in enumerate(episode_data):
        info = step.get("info", {})
        if isinstance(info, dict) and info.get("intervention", False):
            return float(i)
        # Also check direct flags
        if step.get("intervention", False) or step.get("fallen", False):
            return float(i)
    return None


def compute_avg_fail_risk(episode_data: List[Dict[str, Any]]) -> float:
    """
    Compute average fail-risk over episode.
    
    Args:
        episode_data: List of step records with 'fail_risk' in info or step data
    
    Returns:
        Average fail-risk
    """
    fail_risks = []
    for step in episode_data:
        info = step.get("info", {})
        if isinstance(info, dict):
            fail_risk = info.get("fail_risk", 0.0)
        else:
            fail_risk = step.get("fail_risk", 0.0)
        fail_risks.append(float(fail_risk))
    
    return float(np.mean(fail_risks)) if fail_risks else 0.0


def compute_near_fail_metrics(episode_data: List[Dict[str, Any]], threshold: float = 0.6) -> Dict[str, float]:
    """
    Compute near-fail metrics.
    
    Args:
        episode_data: List of step records
        threshold: Fail-risk threshold for "near-fail" (default: 0.6)
    
    Returns:
        Dict with:
            - near_fail_steps: Count of steps with fail_risk > threshold
            - near_fail_density: near_fail_steps / episode_length
    """
    near_fail_steps = 0
    episode_length = len(episode_data)
    
    for step in episode_data:
        info = step.get("info", {})
        if isinstance(info, dict):
            fail_risk = info.get("fail_risk", 0.0)
        else:
            fail_risk = step.get("fail_risk", 0.0)
        
        if fail_risk > threshold:
            near_fail_steps += 1
    
    near_fail_density = float(near_fail_steps / episode_length) if episode_length > 0 else 0.0
    
    return {
        "near_fail_steps": float(near_fail_steps),
        "near_fail_density": near_fail_density
    }


def compute_recovery_success_rate(episode_data: List[Dict[str, Any]]) -> float:
    """
    Compute recovery success rate.
    
    Recovery is considered successful if:
    - Episode entered recovery phase
    - Episode did not end with intervention/fall
    
    Args:
        episode_data: List of step records
    
    Returns:
        Recovery success rate (0.0 to 1.0)
    """
    entered_recovery = False
    recovery_ended_well = True
    
    for step in episode_data:
        info = step.get("info", {})
        core_state = step.get("core_state", {})
        
        # Check if entered recovery
        phase = core_state.get("phase", "stable") if isinstance(core_state, dict) else "stable"
        if phase == "recovery":
            entered_recovery = True
        
        # Check if ended with intervention
        if isinstance(info, dict):
            intervention = info.get("intervention", False)
            fallen = info.get("fallen", False)
        else:
            intervention = step.get("intervention", False)
            fallen = step.get("fallen", False)
        
        if intervention or fallen:
            recovery_ended_well = False
    
    if not entered_recovery:
        return 1.0  # No recovery needed = success
    
    return 1.0 if recovery_ended_well else 0.0


def compute_episode_score_v8(summary: Dict[str, Any]) -> float:
    """
    Compute EDON Score v8 from episode summary.
    
    v8 score explicitly includes:
    - Time-to-intervention (longer is better)
    - Near-fail density (lower is better)
    - Stability (lower is better)
    - Length (longer is better, up to a point)
    
    Args:
        summary: Episode summary dict with:
            - interventions (or interventions_per_episode)
            - stability_score (or stability_avg)
            - episode_length (or avg_episode_length)
            - time_to_first_intervention (optional)
            - near_fail_density (optional)
    
    Returns:
        EDON v8 score (float, typically in range [0, 100+])
    """
    # Extract metrics
    interventions = summary.get("interventions_per_episode") or summary.get("interventions", 0)
    stability = summary.get("stability_avg") or summary.get("stability_score", 0.0)
    episode_length = summary.get("avg_episode_length") or summary.get("episode_length", 0)
    time_to_intervention = summary.get("time_to_first_intervention", None)
    near_fail_density = summary.get("near_fail_density", 0.0)
    
    # Intervention score (fewer is better)
    intervention_score = max(0.0, 100.0 - (interventions * 2.0))
    
    # Stability score (lower is better)
    stability_score = 100.0 * (1.0 - min(1.0, stability * 10.0))
    
    # Time-to-intervention score (longer is better)
    if time_to_intervention is not None:
        # Normalize: 100 steps = 10 points, 200 steps = 15 points, 300+ steps = 20 points
        time_score = min(20.0, time_to_intervention / 10.0)
    else:
        # No intervention = full score
        time_score = 20.0
    
    # Near-fail density score (lower is better)
    # 0.0 density = 10 points, 0.5 density = 5 points, 1.0 density = 0 points
    near_fail_score = 10.0 * (1.0 - near_fail_density)
    
    # Length bonus (longer is better, up to a point)
    length_bonus = min(20.0, episode_length / 50.0)
    
    # Weighted combination
    # Interventions and stability are primary (35% each)
    # Time-to-intervention and near-fail are secondary (10% each)
    # Length is tertiary (10%)
    total_score = (
        0.35 * intervention_score +
        0.35 * stability_score +
        0.10 * time_score +
        0.10 * near_fail_score +
        0.10 * length_bonus
    )
    
    return float(total_score)


def compute_episode_metrics_v8(episode_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute all v8 metrics for an episode.
    
    Args:
        episode_data: List of step records
    
    Returns:
        Dict with all v8 metrics
    """
    # Basic metrics
    # Check for interventions: info["intervention"], info["fallen"], or direct step fields
    interventions = sum(1 for step in episode_data if (
        (step.get("info", {}).get("intervention", False) if isinstance(step.get("info"), dict) else False) or
        (step.get("info", {}).get("fallen", False) if isinstance(step.get("info"), dict) else False) or
        step.get("intervention", False) or 
        step.get("fallen", False)
    ))
    
    episode_length = len(episode_data)
    
    # Stability (from observations) - match baseline computation (variance, not mean absolute)
    roll_list = []
    pitch_list = []
    for step in episode_data:
        obs = step.get("obs")
        if obs is None:
            obs = step.get("state")
        if obs is None:
            obs = step
        if isinstance(obs, dict):
            roll = obs.get("roll", 0.0)  # Don't take abs - need raw values for variance
            pitch = obs.get("pitch", 0.0)
            roll_list.append(float(roll))
            pitch_list.append(float(pitch))
        elif isinstance(obs, (list, np.ndarray)):
            # If obs is an array, skip stability computation for this step
            continue
    
    # Compute stability as variance (matching baseline: var(roll) + var(pitch))
    if len(roll_list) > 1 and len(pitch_list) > 1:
        var_roll = float(np.var(roll_list))
        var_pitch = float(np.var(pitch_list))
        stability = var_roll + var_pitch
    elif len(roll_list) > 0 or len(pitch_list) > 0:
        stability = 0.0  # Single value = no variance
    else:
        stability = 0.0  # No data
    
    # v8-specific metrics
    time_to_intervention = compute_time_to_first_intervention(episode_data)
    avg_fail_risk = compute_avg_fail_risk(episode_data)
    near_fail_metrics = compute_near_fail_metrics(episode_data)
    recovery_success_rate = compute_recovery_success_rate(episode_data)
    
    # Build summary
    summary = {
        "interventions": interventions,
        "interventions_per_episode": float(interventions),
        "stability_score": stability,
        "stability_avg": stability,
        "episode_length": episode_length,
        "avg_episode_length": float(episode_length),
        "time_to_first_intervention": time_to_intervention,
        "avg_fail_risk": avg_fail_risk,
        "near_fail_steps": near_fail_metrics["near_fail_steps"],
        "near_fail_density": near_fail_metrics["near_fail_density"],
        "recovery_success_rate": recovery_success_rate
    }
    
    # Compute v8 score
    summary["edon_score_v8"] = compute_episode_score_v8(summary)
    
    return summary

