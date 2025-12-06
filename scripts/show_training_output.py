"""
Show Training Output

Quick script to show the latest training output by tailing log files.
"""

import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def show_training_output():
    """Show latest training output."""
    log_dir = Path("training")
    log_files = sorted(
        log_dir.glob("edon_train_high_stress_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    print("="*80)
    print("EDON v8 Temporal Memory Training - Latest Output")
    print("="*80)
    
    if not log_files:
        print("No log files found yet. Training may still be initializing...")
        return
    
    latest_log = log_files[0]
    print(f"\nReading from: {latest_log.name}")
    print(f"Last modified: {datetime.fromtimestamp(latest_log.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*80)
    
    # Read last 30 lines
    try:
        with open(latest_log, 'r') as f:
            lines = f.readlines()
            if lines:
                print(f"\nLast {min(30, len(lines))} log entries:\n")
                for line in lines[-30:]:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            episode = data.get("episode")
                            if episode is not None:
                                score = data.get("score", 0.0)
                                reward = data.get("reward", 0.0)
                                length = data.get("length", 0)
                                interventions = data.get("interventions", 0)
                                time_to_int = data.get("time_to_first_intervention")
                                
                                time_str = f"{time_to_int:.0f}" if time_to_int is not None else "N/A"
                                print(
                                    f"Ep {episode:3d} | "
                                    f"Score: {score:5.2f} | "
                                    f"Reward: {reward:7.2f} | "
                                    f"Len: {length:3d} | "
                                    f"Int: {interventions:2d} | "
                                    f"TTI: {time_str:>4s}"
                                )
                        except:
                            # If not JSON, print as-is
                            if "V8" in line or "ep=" in line or "Saving" in line:
                                print(line.strip())
            else:
                print("Log file is empty")
    except Exception as e:
        print(f"Error reading log: {e}")
    
    print("\n" + "="*80)
    print("To see live updates, run: python scripts/monitor_training_live_temporal.py")
    print("="*80)


if __name__ == "__main__":
    show_training_output()

