"""Run v8 training with live output"""
import sys
import subprocess
import os

# Change to the script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("Starting EDON v8 Strategy Policy Training")
print("=" * 70)
print("\nTraining Configuration:")
print("  Episodes: 300")
print("  Profile: high_stress")
print("  Model: edon_v8_strategy_memory_features")
print("  Features: Temporal memory (8 frames) + Early-warning features")
print("\n" + "=" * 70)
print("Live training output:")
print("=" * 70 + "\n")

# Run training with unbuffered output
cmd = [
    sys.executable, "-u",
    "training/train_edon_v8_strategy.py",
    "--episodes", "300",
    "--profile", "high_stress",
    "--seed", "0",
    "--lr", "5e-4",
    "--gamma", "0.995",
    "--update-epochs", "10",
    "--output-dir", "models",
    "--model-name", "edon_v8_strategy_memory_features",
    "--fail-risk-model", "models/edon_fail_risk_v1_fixed_v2.pt",
    "--max-steps", "1000",
    "--w-intervention", "20.0",
    "--w-stability", "1.0",
    "--w-torque", "0.1"
]

# Run with live output
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
    bufsize=1
)

# Print output line by line
try:
    for line in process.stdout:
        print(line, end='', flush=True)
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user")
    process.terminate()
    sys.exit(1)

process.wait()
print(f"\n\nTraining completed with exit code: {process.returncode}")

