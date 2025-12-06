"""
Check Training Progress

Quick script to check how many episodes have been completed.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_progress():
    """Check training progress."""
    model_name = "edon_v8_strategy_temporal_v1"
    model_path = Path(f"models/{model_name}.pt")
    log_dir = Path("training")
    total_episodes = 300
    
    # Find latest log file
    log_files = sorted(
        log_dir.glob("edon_train_high_stress_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    episodes_completed = 0
    latest_episode_data = None
    
    if log_files:
        latest_log = log_files[0]
        try:
            with open(latest_log, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            episode = data.get("episode")
                            if episode is not None:
                                episodes_completed = max(episodes_completed, episode + 1)
                                if episode == episodes_completed - 1:
                                    latest_episode_data = data
                        except:
                            pass
        except Exception as e:
            print(f"Error reading log: {e}")
    
    progress_pct = (episodes_completed / total_episodes * 100) if total_episodes > 0 else 0
    
    print("="*80)
    print("Training Progress Check")
    print("="*80)
    print(f"Episodes completed: {episodes_completed}/{total_episodes}")
    print(f"Progress: {progress_pct:.1f}%")
    print(f"Remaining: {total_episodes - episodes_completed} episodes")
    
    if model_path.exists():
        mtime = model_path.stat().st_mtime
        age_min = (time.time() - mtime) / 60
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"\nModel checkpoint: Yes")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Last updated: {age_min:.1f} minutes ago")
    else:
        print(f"\nModel checkpoint: Not yet created")
    
    if latest_episode_data:
        print(f"\nLatest episode ({episodes_completed - 1}):")
        print(f"  Score: {latest_episode_data.get('score', 0.0):.2f}")
        print(f"  Reward: {latest_episode_data.get('reward', 0.0):.2f}")
        print(f"  Length: {latest_episode_data.get('length', 0)}")
        print(f"  Interventions: {latest_episode_data.get('interventions', 0)}")
    
    print("="*80)
    
    return progress_pct


if __name__ == "__main__":
    check_progress()

