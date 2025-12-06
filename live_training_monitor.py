"""
Live training monitor for EDON v7.
Shows real-time progress, episode count, and estimated time remaining.
"""
import os
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

def get_model_info(model_path):
    """Get model file info."""
    if not model_path.exists():
        return None
    
    stat = model_path.stat()
    return {
        "exists": True,
        "size_mb": stat.st_size / (1024 * 1024),
        "mtime": datetime.fromtimestamp(stat.st_mtime),
        "age_seconds": time.time() - stat.st_mtime
    }

def count_episodes_from_logs():
    """Count total episodes from all recent log files."""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return 0
    
    total_episodes = 0
    for log_file in logs_dir.glob("edon_train_*.jsonl"):
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            if record.get('type') == 'episode_summary':
                                total_episodes += 1
                        except:
                            pass
        except:
            pass
    
    return total_episodes

def estimate_episodes_from_model(model_path, target_episodes=300):
    """Try to estimate current episode from model checkpoint timing."""
    if not model_path.exists():
        return None
    
    # Model was last saved at 21:14:55
    # If training started around 19:04 (from latest log), that's ~2h 10min for some episodes
    # But we need to track when training actually started
    # For now, use file modification time as a proxy
    stat = model_path.stat()
    model_age = time.time() - stat.st_mtime
    
    # If model was just updated, training is active
    # We can't know exact episode without parsing stdout, but we can estimate
    return None  # Can't reliably estimate without more info

def monitor_training_live(model_name="edon_v7_ep300_aligned.pt", target_episodes=300, refresh_interval=2):
    """
    Monitor training in real-time with live updates.
    
    Args:
        model_name: Name of model file to monitor
        target_episodes: Target number of episodes (default 300)
        refresh_interval: Seconds between updates
    """
    model_path = Path("models") / model_name
    
    print("="*80)
    print("EDON v7 LIVE TRAINING MONITOR")
    print("="*80)
    print(f"Monitoring: {model_name}")
    print(f"Target episodes: {target_episodes}")
    print(f"Refresh interval: {refresh_interval}s")
    print("="*80)
    print("\nPress Ctrl+C to stop monitoring\n")
    
    start_time = time.time()
    last_model_mtime = 0
    last_episode_count = 0
    episode_times = []  # Track time between episodes
    
    try:
        while True:
            # Clear screen (works in most terminals)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            current_time = datetime.now()
            elapsed = time.time() - start_time
            
            # Check model file
            model_info = get_model_info(model_path)
            
            # Count episodes from logs
            current_episodes = count_episodes_from_logs()
            
            # Check if model was updated
            model_updated = False
            if model_info and model_info["mtime"].timestamp() > last_model_mtime:
                model_updated = True
                last_model_mtime = model_info["mtime"].timestamp()
            
            # Estimate progress
            if current_episodes > 0:
                progress_pct = (current_episodes / target_episodes) * 100
                remaining_episodes = max(0, target_episodes - current_episodes)
            else:
                progress_pct = 0
                remaining_episodes = target_episodes
            
            # Calculate time estimates
            if current_episodes > last_episode_count and elapsed > 0:
                # New episodes completed
                episodes_since_last = current_episodes - last_episode_count
                time_per_episode = elapsed / current_episodes if current_episodes > 0 else 0
                episode_times.append(time_per_episode)
                if len(episode_times) > 10:
                    episode_times.pop(0)  # Keep last 10
                
                last_episode_count = current_episodes
            
            # Estimate time remaining
            if len(episode_times) > 0 and remaining_episodes > 0:
                avg_time_per_episode = sum(episode_times) / len(episode_times)
                estimated_seconds_remaining = avg_time_per_episode * remaining_episodes
                eta = datetime.now() + timedelta(seconds=estimated_seconds_remaining)
            else:
                estimated_seconds_remaining = None
                eta = None
            
            # Display status
            print("="*80)
            print(f"EDON v7 LIVE TRAINING MONITOR - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            print()
            
            # Model status
            if model_info:
                status_icon = "[OK]" if model_updated else "[  ]"
                print(f"{status_icon} Model: {model_name}")
                print(f"   Size: {model_info['size_mb']:.2f} MB")
                print(f"   Last updated: {model_info['mtime'].strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Age: {model_info['age_seconds']:.0f} seconds ago")
                if model_updated:
                    print("   *** MODEL JUST UPDATED! ***")
            else:
                print("[  ] Model: Not found yet")
            print()
            
            # Progress
            print(f"[PROGRESS] {current_episodes}/{target_episodes} episodes ({progress_pct:.1f}%)")
            if remaining_episodes > 0:
                print(f"   Remaining: {remaining_episodes} episodes")
            else:
                print("   *** Training complete! ***")
            print()
            
            # Time estimates
            print(f"[TIME] Elapsed: {timedelta(seconds=int(elapsed))}")
            if estimated_seconds_remaining is not None and estimated_seconds_remaining > 0:
                print(f"   Estimated remaining: {timedelta(seconds=int(estimated_seconds_remaining))}")
                print(f"   ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
                if len(episode_times) > 0:
                    avg_ep_time = sum(episode_times) / len(episode_times)
                    print(f"   Avg time/episode: {avg_ep_time:.1f}s")
            else:
                print("   Waiting for episode data...")
            print()
            
            # Recent log files
            logs_dir = Path("logs")
            if logs_dir.exists():
                log_files = sorted(logs_dir.glob("edon_train_*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)
                if log_files:
                    latest_log = log_files[0]
                    log_mtime = datetime.fromtimestamp(latest_log.stat().st_mtime)
                    print(f"[LOG] Latest: {latest_log.name}")
                    print(f"   Updated: {log_mtime.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"   Age: {(time.time() - log_mtime.timestamp()):.0f} seconds ago")
            print()
            
            # Process status
            try:
                import subprocess
                result = subprocess.run(
                    ["powershell", "-Command", "Get-Process python -ErrorAction SilentlyContinue | Measure-Object | Select-Object -ExpandProperty Count"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    proc_count = int(result.stdout.strip())
                    status = "[ACTIVE]" if proc_count > 0 else "[STOPPED]"
                    print(f"[PROCESS] {status} ({proc_count} Python process(es))")
            except:
                print("[PROCESS] Unable to check")
            print()
            
            print("="*80)
            print(f"Refreshing every {refresh_interval}s... (Press Ctrl+C to stop)")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("Monitoring stopped by user")
        print("="*80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Live EDON v7 training monitor")
    parser.add_argument("--model-name", type=str, default="edon_v7_ep300_aligned.pt",
                       help="Model filename to monitor")
    parser.add_argument("--target-episodes", type=int, default=300,
                       help="Target number of episodes")
    parser.add_argument("--refresh", type=float, default=2.0,
                       help="Refresh interval in seconds")
    
    args = parser.parse_args()
    
    monitor_training_live(
        model_name=args.model_name,
        target_episodes=args.target_episodes,
        refresh_interval=args.refresh
    )

