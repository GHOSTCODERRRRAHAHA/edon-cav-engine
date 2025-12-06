"""
Wait for training to complete and then run evaluation.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def wait_for_model(model_path: str, timeout_minutes: int = 120, check_interval: int = 30):
    """Wait for model file to be created."""
    model_file = Path(model_path)
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    
    print(f"Waiting for model: {model_path}")
    print(f"Timeout: {timeout_minutes} minutes")
    print(f"Check interval: {check_interval} seconds")
    print()
    
    while True:
        if model_file.exists():
            # Check if file was modified recently (within last 10 seconds)
            # This indicates training just finished
            file_age = time.time() - model_file.stat().st_mtime
            if file_age < 10:
                print(f"Model found! (created {file_age:.1f} seconds ago)")
                return True
        
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            print(f"Timeout reached ({timeout_minutes} minutes)")
            return False
        
        print(f"Waiting... ({elapsed/60:.1f} minutes elapsed)", end='\r')
        time.sleep(check_interval)
    
    return False

if __name__ == "__main__":
    model_path = "models/edon_v8_strategy_intervention_first.pt"
    if wait_for_model(model_path, timeout_minutes=120):
        print("\nTraining complete! Ready for evaluation.")
    else:
        print("\nTimeout - training may still be running. Check manually.")

