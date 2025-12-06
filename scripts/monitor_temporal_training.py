"""
Monitor Temporal Memory Training Progress

Quick script to check training progress for the new temporal memory policy.
"""

import sys
from pathlib import Path
import time
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

def monitor_training():
    """Monitor training progress."""
    model_path = Path("models/edon_v8_strategy_temporal_v1.pt")
    
    print("="*80)
    print("EDON v8 Temporal Memory Training Monitor")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if model_path.exists():
        mtime = model_path.stat().st_mtime
        age_minutes = (time.time() - mtime) / 60
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"[OK] Model checkpoint found: {model_path.name}")
        print(f"     Last modified: {datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"     Age: {age_minutes:.1f} minutes ago")
        print(f"     Size: {size_mb:.2f} MB")
    else:
        print(f"[WAIT] Model not yet created: {model_path.name}")
    
    # Check for log files
    log_dir = Path("training")
    log_files = sorted(log_dir.glob("edon_train_high_stress_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if log_files:
        latest_log = log_files[0]
        mtime = latest_log.stat().st_mtime
        age_minutes = (time.time() - mtime) / 60
        
        # Count episodes in log
        episode_count = 0
        try:
            import json
            with open(latest_log) as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if data.get("episode") is not None:
                                episode_count = max(episode_count, data.get("episode", 0) + 1)
                        except:
                            pass
        except:
            pass
        
        print(f"\n[OK] Latest log: {latest_log.name}")
        print(f"     Last modified: {datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"     Age: {age_minutes:.1f} minutes ago")
        print(f"     Episodes logged: {episode_count}")
    else:
        print("\n[WAIT] No log files found yet")
    
    print("\n" + "="*80)
    print("Training Status: ACTIVE")
    print("="*80)
    print("\nNew features in this training:")
    print("  - Early-warning features (rolling variance, oscillation energy, near-fail density)")
    print("  - Stacked observations (8 frames for temporal memory)")
    print("  - Input size: 248 (31 base features × 8 frames)")
    print("\nTo view real-time output, check the terminal where training was started.")
    print("="*80)


if __name__ == "__main__":
    monitor_training()

