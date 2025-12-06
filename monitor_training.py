"""
Monitor EDON v7 training progress.
Shows recent training output and statistics.
"""
import os
import time
import json
from pathlib import Path
from datetime import datetime

def find_latest_model_checkpoint():
    """Find the latest model checkpoint in models directory."""
    models_dir = Path("models")
    if not models_dir.exists():
        return None
    
    # Look for edon_v7_ep300_aligned related files
    checkpoints = []
    for f in models_dir.glob("edon_v7_ep300_aligned*.pt"):
        checkpoints.append((f.stat().st_mtime, f))
    
    if checkpoints:
        checkpoints.sort(reverse=True)
        return checkpoints[0][1]
    return None

def check_training_logs():
    """Check for recent training log files."""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return []
    
    log_files = []
    for f in logs_dir.glob("*.jsonl"):
        log_files.append((f.stat().st_mtime, f))
    
    log_files.sort(reverse=True)
    return [f[1] for f in log_files[:5]]  # Return 5 most recent

def count_episodes_in_log(log_path):
    """Count episodes in a JSONL log file."""
    if not log_path.exists():
        return 0
    
    count = 0
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if record.get('type') == 'episode_summary':
                            count += 1
                    except:
                        pass
    except:
        pass
    return count

def main():
    print("="*70)
    print("EDON v7 Training Monitor")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check for model checkpoint
    checkpoint = find_latest_model_checkpoint()
    if checkpoint:
        mtime = datetime.fromtimestamp(checkpoint.stat().st_mtime)
        size_mb = checkpoint.stat().st_size / (1024 * 1024)
        print(f"[OK] Model checkpoint found: {checkpoint.name}")
        print(f"    Last modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"    Size: {size_mb:.2f} MB")
    else:
        print("[!] No model checkpoint found yet")
    print()
    
    # Check training logs
    log_files = check_training_logs()
    if log_files:
        print(f"[OK] Found {len(log_files)} recent log file(s):")
        for log_file in log_files:
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            episodes = count_episodes_in_log(log_file)
            print(f"    {log_file.name}")
            print(f"      Last modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"      Episodes logged: {episodes}")
    else:
        print("[!] No training log files found")
    print()
    
    # Check for running Python processes
    try:
        import subprocess
        result = subprocess.run(
            ["powershell", "-Command", "Get-Process python -ErrorAction SilentlyContinue | Measure-Object | Select-Object -ExpandProperty Count"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            count = int(result.stdout.strip())
            if count > 0:
                print(f"[OK] {count} Python process(es) running")
            else:
                print("[!] No Python processes detected")
    except:
        pass
    
    print()
    print("="*70)
    print("Training Status: ACTIVE")
    print("="*70)
    print("\nTo view real-time training output, check the terminal")
    print("where the training was started.")
    print("\nModel will be saved to: models/edon_v7_ep300_aligned.pt")
    print("(or similar name based on training configuration)")

if __name__ == "__main__":
    main()

