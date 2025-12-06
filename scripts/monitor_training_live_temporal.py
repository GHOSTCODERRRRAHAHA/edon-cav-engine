"""
Live Training Monitor for Temporal Memory Policy

Monitors training progress in real-time by watching log files and model checkpoints.
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))


def monitor_live():
    """Monitor training in real-time."""
    model_name = "edon_v8_strategy_temporal_v1"
    model_path = Path(f"models/{model_name}.pt")
    log_dir = Path("training")
    
    print("="*80)
    print("EDON v8 Temporal Memory Training - Live Monitor")
    print("="*80)
    print(f"Model: {model_name}.pt")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print("\nWatching for updates... (Press Ctrl+C to stop)\n")
    
    last_episode = -1
    last_model_mtime = 0
    episode_stats = deque(maxlen=10)  # Last 10 episodes
    
    try:
        while True:
            # Check model file
            if model_path.exists():
                current_mtime = model_path.stat().st_mtime
                if current_mtime > last_model_mtime:
                    last_model_mtime = current_mtime
                    size_mb = model_path.stat().st_size / (1024 * 1024)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Model updated: {size_mb:.2f} MB")
            
            # Check log files
            log_files = sorted(
                log_dir.glob("edon_train_high_stress_*.jsonl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            if log_files:
                latest_log = log_files[0]
                
                # Read latest episodes
                try:
                    with open(latest_log, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            # Get last few lines
                            for line in lines[-20:]:
                                if line.strip():
                                    try:
                                        data = json.loads(line)
                                        episode = data.get("episode")
                                        if episode is not None and episode > last_episode:
                                            last_episode = episode
                                            
                                            # Extract metrics
                                            score = data.get("score", 0.0)
                                            reward = data.get("reward", 0.0)
                                            length = data.get("length", 0)
                                            interventions = data.get("interventions", 0)
                                            time_to_int = data.get("time_to_first_intervention")
                                            near_fail = data.get("near_fail_density", 0.0)
                                            
                                            episode_stats.append({
                                                "episode": episode,
                                                "score": score,
                                                "reward": reward,
                                                "length": length,
                                                "interventions": interventions,
                                                "time_to_int": time_to_int,
                                                "near_fail": near_fail
                                            })
                                            
                                            # Print latest episode
                                            time_str = f"{time_to_int:.0f}" if time_to_int is not None else "N/A"
                                            print(
                                                f"[{datetime.now().strftime('%H:%M:%S')}] "
                                                f"Ep {episode:3d} | "
                                                f"Score: {score:5.2f} | "
                                                f"Reward: {reward:7.2f} | "
                                                f"Len: {length:3d} | "
                                                f"Int: {interventions:2d} | "
                                                f"TTI: {time_str:>4s} | "
                                                f"NF: {near_fail:.4f}"
                                            )
                                            
                                            # Print summary every 10 episodes
                                            if episode % 10 == 0 and len(episode_stats) >= 10:
                                                avg_score = sum(s["score"] for s in episode_stats) / len(episode_stats)
                                                avg_reward = sum(s["reward"] for s in episode_stats) / len(episode_stats)
                                                avg_length = sum(s["length"] for s in episode_stats) / len(episode_stats)
                                                avg_int = sum(s["interventions"] for s in episode_stats) / len(episode_stats)
                                                
                                                print(
                                                    f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                                                    f"SUMMARY (last 10): "
                                                    f"Avg Score: {avg_score:.2f} | "
                                                    f"Avg Reward: {avg_reward:.2f} | "
                                                    f"Avg Length: {avg_length:.1f} | "
                                                    f"Avg Int: {avg_int:.2f}\n"
                                                )
                                    except (json.JSONDecodeError, KeyError):
                                        pass
                except Exception as e:
                    pass
            
            time.sleep(2)  # Check every 2 seconds
            
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("Monitoring stopped")
        print("="*80)
        if episode_stats:
            print(f"\nLast {len(episode_stats)} episodes:")
            for stat in list(episode_stats)[-5:]:
                print(f"  Ep {stat['episode']}: Score={stat['score']:.2f}, Int={stat['interventions']}, Reward={stat['reward']:.2f}")


if __name__ == "__main__":
    monitor_live()

