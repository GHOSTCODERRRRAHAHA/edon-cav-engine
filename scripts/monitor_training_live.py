"""
Live Training Monitor for v8 Strategy Policy

Shows real-time training progress by monitoring:
- Model file updates
- Training log files
- Episode progress
"""

import time
import json
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

def monitor_training(model_name: str = "edon_v8_strategy_v1_no_reflex", check_interval: int = 5):
    """Monitor training progress in real-time."""
    model_path = Path(f"models/{model_name}.pt")
    logs_dir = Path("logs")
    
    print("="*80)
    print("Live Training Monitor")
    print("="*80)
    print(f"Monitoring: {model_name}")
    print(f"Check interval: {check_interval} seconds")
    print("Press Ctrl+C to stop")
    print("="*80)
    print()
    
    last_model_mtime = 0
    last_episode_count = 0
    episode_count = 0
    
    try:
        while True:
            # Check model file
            if model_path.exists():
                current_mtime = model_path.stat().st_mtime
                age_seconds = time.time() - current_mtime
                size_kb = model_path.stat().st_size / 1024
                
                if current_mtime != last_model_mtime:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Model updated!")
                    print(f"  File: {model_path.name}")
                    print(f"  Size: {size_kb:.1f} KB")
                    print(f"  Age: {age_seconds:.0f} seconds ago")
                    
                    # Try to load checkpoint info
                    try:
                        import torch
                        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                        episodes = checkpoint.get("episodes", "unknown")
                        final_score = checkpoint.get("final_avg_score", "unknown")
                        print(f"  Episodes: {episodes}")
                        print(f"  Final avg score: {final_score:.2f}" if isinstance(final_score, (int, float)) else f"  Final avg score: {final_score}")
                    except:
                        pass
                    
                    print()
                    last_model_mtime = current_mtime
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Model age: {age_seconds:.0f}s (no update)")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Model not created yet...")
            
            # Check for training log files
            if logs_dir.exists():
                log_files = sorted(logs_dir.glob("edon_train_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
                if log_files:
                    latest_log = log_files[0]
                    log_mtime = latest_log.stat().st_mtime
                    log_age = time.time() - log_mtime
                    
                    # Count episodes in latest log
                    try:
                        with open(latest_log, 'r') as f:
                            lines = f.readlines()
                            current_episode_count = sum(1 for line in lines if line.strip() and json.loads(line).get("type") == "episode_summary")
                        
                        if current_episode_count != last_episode_count:
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] New episode logged!")
                            print(f"  Log: {latest_log.name}")
                            print(f"  Episodes: {current_episode_count}")
                            
                            # Show latest episode summary if available
                            try:
                                for line in reversed(lines):
                                    if line.strip():
                                        data = json.loads(line)
                                        if data.get("type") == "episode_summary":
                                            ep_id = data.get("episode_id", "unknown")
                                            score = data.get("edon_score", data.get("score", "unknown"))
                                            interventions = data.get("interventions", "unknown")
                                            stability = data.get("stability_score", "unknown")
                                            print(f"  Latest episode {ep_id}: score={score}, interventions={interventions}, stability={stability:.4f}" if isinstance(stability, (int, float)) else f"  Latest episode {ep_id}: score={score}, interventions={interventions}, stability={stability}")
                                            break
                            except:
                                pass
                            
                            print()
                            last_episode_count = current_episode_count
                    except:
                        pass
            
            # Check Python processes
            try:
                import subprocess
                result = subprocess.run(["tasklist"], capture_output=True, text=True)
                python_count = result.stdout.count("python.exe")
                if python_count > 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Python processes: {python_count} (training likely active)")
            except:
                pass
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print()
        print("="*80)
        print("Monitoring stopped")
        print("="*80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="edon_v8_strategy_v1_no_reflex", help="Model name to monitor")
    parser.add_argument("--interval", type=int, default=5, help="Check interval in seconds")
    args = parser.parse_args()
    
    monitor_training(model_name=args.model_name, check_interval=args.interval)

