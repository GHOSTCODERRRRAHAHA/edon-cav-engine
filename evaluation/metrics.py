"""Metrics tracking and computation for evaluation."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
import json
from pathlib import Path


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    
    episode_id: int
    interventions: int = 0
    freeze_events: int = 0
    stability_score: float = 0.0
    episode_length: int = 0  # steps
    episode_time: float = 0.0  # seconds
    success: bool = False
    
    # Tilt zone tracking (new metrics for prefall analysis)
    prefall_events: int = 0  # Count of steps spent in prefall zone
    prefall_time: float = 0.0  # Total time in prefall zone (seconds)
    prefall_times: List[float] = field(default_factory=list)  # Timestamps of prefall events
    safe_time: float = 0.0  # Time spent in safe zone
    fail_events: int = 0  # Count of actual failures (interventions)
    
    # Detailed tracking
    roll_history: List[float] = field(default_factory=list)
    pitch_history: List[float] = field(default_factory=list)
    com_history: List[float] = field(default_factory=list)  # center of mass positions
    intervention_times: List[float] = field(default_factory=list)
    freeze_times: List[float] = field(default_factory=list)
    tilt_zone_history: List[str] = field(default_factory=list)  # "safe", "prefall", "fail" per step
    
    # Enhanced stability metrics (computed after episode)
    roll_rms: float = 0.0
    pitch_rms: float = 0.0
    roll_max: float = 0.0
    pitch_max: float = 0.0
    roll_std: float = 0.0
    pitch_std: float = 0.0
    com_deviation: float = 0.0  # RMS of COM position
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        # Convert numpy types to native Python types
        def convert_value(v):
            if isinstance(v, (np.integer, np.floating)):
                return float(v) if isinstance(v, np.floating) else int(v)
            elif isinstance(v, np.bool_):
                return bool(v)
            elif isinstance(v, (list, tuple)):
                return [convert_value(item) for item in v]
            elif isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            return v
        
        return {
            "episode_id": int(self.episode_id),
            "interventions": int(self.interventions),
            "freeze_events": int(self.freeze_events),
            "stability_score": float(self.stability_score),
            "episode_length": int(self.episode_length),
            "episode_time": float(self.episode_time),
            "success": bool(self.success),
            "roll_rms": float(self.roll_rms),
            "pitch_rms": float(self.pitch_rms),
            "roll_max": float(self.roll_max),
            "pitch_max": float(self.pitch_max),
            "roll_std": float(self.roll_std),
            "pitch_std": float(self.pitch_std),
            "com_deviation": float(self.com_deviation),
            "prefall_events": int(self.prefall_events),
            "prefall_time": float(self.prefall_time),
            "safe_time": float(self.safe_time),
            "fail_events": int(self.fail_events),
            "intervention_times": [float(t) for t in self.intervention_times],
            "freeze_times": [float(t) for t in self.freeze_times],
            "prefall_times": [float(t) for t in self.prefall_times],
            "metadata": convert_value(self.metadata)
        }


@dataclass
class RunMetrics:
    """Aggregated metrics for a full evaluation run."""
    
    mode: str  # "baseline" or "edon"
    episodes: int
    interventions_total: int
    interventions_per_episode: float
    freeze_events_total: int
    freeze_events_per_episode: float
    stability_avg: float
    stability_std: float
    stability_min: float
    stability_max: float
    success_rate: float
    avg_episode_length: float
    avg_episode_time: float
    
    # Enhanced stability metrics (averages across episodes)
    roll_rms_avg: float = 0.0
    pitch_rms_avg: float = 0.0
    roll_max_avg: float = 0.0
    pitch_max_avg: float = 0.0
    com_deviation_avg: float = 0.0
    
    # Tilt zone metrics (averages across episodes)
    prefall_events_total: int = 0
    prefall_events_per_episode: float = 0.0
    prefall_time_avg: float = 0.0
    safe_time_avg: float = 0.0
    fail_events_total: int = 0
    fail_events_per_episode: float = 0.0
    
    # Raw episode metrics
    raw_episode_metrics: List[EpisodeMetrics] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        # Convert numpy types to native Python types
        result = {
            "mode": str(self.mode),
            "episodes": int(self.episodes),
            "interventions_total": int(self.interventions_total),
            "interventions_per_episode": float(self.interventions_per_episode),
            "freeze_events_total": int(self.freeze_events_total),
            "freeze_events_per_episode": float(self.freeze_events_per_episode),
            "stability_avg": float(self.stability_avg),
            "stability_std": float(self.stability_std),
            "stability_min": float(self.stability_min),
            "stability_max": float(self.stability_max),
            "success_rate": float(self.success_rate),
            "avg_episode_length": float(self.avg_episode_length),
            "avg_episode_time": float(self.avg_episode_time),
            "roll_rms_avg": float(self.roll_rms_avg),
            "pitch_rms_avg": float(self.pitch_rms_avg),
            "roll_max_avg": float(self.roll_max_avg),
            "pitch_max_avg": float(self.pitch_max_avg),
            "com_deviation_avg": float(self.com_deviation_avg),
            "prefall_events_total": int(self.prefall_events_total),
            "prefall_events_per_episode": float(self.prefall_events_per_episode),
            "prefall_time_avg": float(self.prefall_time_avg),
            "safe_time_avg": float(self.safe_time_avg),
            "fail_events_total": int(self.fail_events_total),
            "fail_events_per_episode": float(self.fail_events_per_episode),
            "raw_episode_metrics": [ep.to_dict() for ep in self.raw_episode_metrics]
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result
    
    def save_json(self, path: Path) -> None:
        """Save metrics to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def save_csv(self, path: Path) -> None:
        """Save per-episode metrics to CSV file."""
        import csv
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                "episode_id", "interventions", "freeze_events", "stability_score",
                "episode_length", "episode_time", "success"
            ])
            # Data rows
            for ep in self.raw_episode_metrics:
                writer.writerow([
                    ep.episode_id, ep.interventions, ep.freeze_events,
                    ep.stability_score, ep.episode_length, ep.episode_time, ep.success
                ])


def compute_stability_score(roll_list: List[float], pitch_list: List[float]) -> float:
    """
    Compute stability score from roll and pitch history.
    
    Lower is better (less variance = more stable).
    
    Args:
        roll_list: List of roll angles (radians)
        pitch_list: List of pitch angles (radians)
    
    Returns:
        Stability score (sum of variances)
    """
    if len(roll_list) == 0 or len(pitch_list) == 0:
        return float('inf')  # No data = unstable
    
    if len(roll_list) != len(pitch_list):
        # Use minimum length
        min_len = min(len(roll_list), len(pitch_list))
        roll_list = roll_list[:min_len]
        pitch_list = pitch_list[:min_len]
    
    var_roll = float(np.var(roll_list)) if len(roll_list) > 1 else 0.0
    var_pitch = float(np.var(pitch_list)) if len(pitch_list) > 1 else 0.0
    
    return var_roll + var_pitch


def aggregate_run_metrics(episodes: List[EpisodeMetrics], mode: str) -> RunMetrics:
    """
    Aggregate episode metrics into run metrics.
    
    Args:
        episodes: List of episode metrics
        mode: "baseline" or "edon"
    
    Returns:
        Aggregated RunMetrics
    """
    if len(episodes) == 0:
        return RunMetrics(
            mode=mode,
            episodes=0,
            interventions_total=0,
            interventions_per_episode=0.0,
            freeze_events_total=0,
            freeze_events_per_episode=0.0,
            stability_avg=0.0,
            stability_std=0.0,
            stability_min=0.0,
            stability_max=0.0,
            success_rate=0.0,
            avg_episode_length=0.0,
            avg_episode_time=0.0,
            raw_episode_metrics=episodes
        )
    
    interventions_total = sum(ep.interventions for ep in episodes)
    freeze_events_total = sum(ep.freeze_events for ep in episodes)
    stability_scores = [ep.stability_score for ep in episodes]
    successes = sum(1 for ep in episodes if ep.success)
    episode_lengths = [ep.episode_length for ep in episodes]
    episode_times = [ep.episode_time for ep in episodes]
    
    # Compute enhanced stability metrics
    roll_rms_list = [ep.roll_rms for ep in episodes if ep.roll_rms > 0]
    pitch_rms_list = [ep.pitch_rms for ep in episodes if ep.pitch_rms > 0]
    roll_max_list = [ep.roll_max for ep in episodes if ep.roll_max > 0]
    pitch_max_list = [ep.pitch_max for ep in episodes if ep.pitch_max > 0]
    com_deviation_list = [ep.com_deviation for ep in episodes if ep.com_deviation > 0]
    
    # Compute tilt zone metrics
    prefall_events_total = sum(ep.prefall_events for ep in episodes)
    prefall_time_list = [ep.prefall_time for ep in episodes]
    safe_time_list = [ep.safe_time for ep in episodes]
    fail_events_total = sum(ep.fail_events for ep in episodes)
    
    return RunMetrics(
        mode=mode,
        episodes=len(episodes),
        interventions_total=interventions_total,
        interventions_per_episode=float(interventions_total / len(episodes)),
        freeze_events_total=freeze_events_total,
        freeze_events_per_episode=float(freeze_events_total / len(episodes)),
        stability_avg=float(np.mean(stability_scores)),
        stability_std=float(np.std(stability_scores)),
        stability_min=float(np.min(stability_scores)),
        stability_max=float(np.max(stability_scores)),
        success_rate=float(successes / len(episodes)),
        avg_episode_length=float(np.mean(episode_lengths)),
        avg_episode_time=float(np.mean(episode_times)),
        roll_rms_avg=float(np.mean(roll_rms_list)) if roll_rms_list else 0.0,
        pitch_rms_avg=float(np.mean(pitch_rms_list)) if pitch_rms_list else 0.0,
        roll_max_avg=float(np.mean(roll_max_list)) if roll_max_list else 0.0,
        pitch_max_avg=float(np.mean(pitch_max_list)) if pitch_max_list else 0.0,
        com_deviation_avg=float(np.mean(com_deviation_list)) if com_deviation_list else 0.0,
        prefall_events_total=prefall_events_total,
        prefall_events_per_episode=float(prefall_events_total / len(episodes)),
        prefall_time_avg=float(np.mean(prefall_time_list)) if prefall_time_list else 0.0,
        safe_time_avg=float(np.mean(safe_time_list)) if safe_time_list else 0.0,
        fail_events_total=fail_events_total,
        fail_events_per_episode=float(fail_events_total / len(episodes)),
        raw_episode_metrics=episodes
    )

