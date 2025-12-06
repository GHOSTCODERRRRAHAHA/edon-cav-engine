"""Check v8 training status."""

from pathlib import Path
import time
import json

# Check model file
model_path = Path("models/edon_v8_strategy_v1_improved.pt")
if model_path.exists():
    mtime = model_path.stat().st_mtime
    age_minutes = (time.time() - mtime) / 60
    size_kb = model_path.stat().st_size / 1024
    print(f"Model file: {model_path.name}")
    print(f"  Last modified: {age_minutes:.1f} minutes ago")
    print(f"  File size: {size_kb:.2f} KB")
    
    # Try to load and check checkpoint info
    try:
        import torch
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        episodes = checkpoint.get("episodes", "unknown")
        final_score = checkpoint.get("final_avg_score", "unknown")
        print(f"  Episodes trained: {episodes}")
        print(f"  Final avg score: {final_score}")
    except:
        pass
else:
    print("Model file not created yet - training still in early stages")

print()

# Check for Python processes
import subprocess
try:
    result = subprocess.run(["tasklist"], capture_output=True, text=True)
    python_count = result.stdout.count("python.exe")
    print(f"Python processes running: {python_count}")
    if python_count > 0:
        print("  (Training likely in progress)")
except:
    pass

print()

# Check logs directory
logs_dir = Path("logs")
if logs_dir.exists():
    log_files = sorted(logs_dir.glob("edon_train_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if log_files:
        latest_log = log_files[0]
        mtime = latest_log.stat().st_mtime
        age_minutes = (time.time() - mtime) / 60
        print(f"Latest training log: {latest_log.name}")
        print(f"  Last modified: {age_minutes:.1f} minutes ago")
        
        # Count episodes in log
        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                episode_count = sum(1 for line in lines if line.strip() and json.loads(line).get("type") == "episode_summary")
                print(f"  Episodes logged: {episode_count}")
        except:
            pass

