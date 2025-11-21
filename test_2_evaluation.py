"""Test 2: Evaluation (WESAD Ground Truth)"""
import subprocess
import sys
import os
from pathlib import Path

print("\n=== TEST 2: Evaluation (WESAD Ground Truth) ===\n")

# Try local path first, then parent folder
wesad_path = Path("data/raw/wesad/wesad_wrist_4hz.csv")
if not wesad_path.exists():
    print(f"Local path not found, trying parent folder...")
    # Try other possible paths
    alt_paths = [
        Path("../data/raw/wesad/wesad_wrist_4hz.csv"),
        Path("../EDON/data/raw/wesad/wesad_wrist_4hz.csv"),
    ]
    found = False
    for alt in alt_paths:
        if alt.exists():
            wesad_path = alt
            found = True
            print(f"Found at: {wesad_path}")
            break
    if not found:
        print("[SKIP] WESAD data not found - skipping evaluation test")
        exit(0)

print(f"Running evaluation on 50 windows (limited for speed)...")
print(f"Data path: {wesad_path}")

result = subprocess.run(
    [sys.executable, "tools/eval_wesad.py", 
     "--data", str(wesad_path),
     "--api", "http://127.0.0.1:8001",
     "--output", "reports/last_eval.json",
     "--limit", "50"],
    capture_output=False,
    text=True
)

if result.returncode == 0:
    print("\n[OK] Evaluation complete")
    eval_file = Path("reports/last_eval.json")
    if eval_file.exists():
        import json
        with open(eval_file) as f:
            eval_data = json.load(f)
        print(f"  Accuracy: {eval_data.get('accuracy', 'N/A')}")
        print(f"  AUROC: {eval_data.get('auroc', 'N/A')}")
    exit(0)
else:
    print(f"\n[FAIL] Evaluation failed with exit code {result.returncode}")
    exit(1)

