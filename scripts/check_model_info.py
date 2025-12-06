"""Check model checkpoint info."""

import torch
from pathlib import Path

model_path = Path("models/edon_v8_strategy_v1_fixed.pt")
if model_path.exists():
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    print("Model Checkpoint Info:")
    print(f"  Episodes: {checkpoint.get('episodes', 'unknown')}")
    print(f"  Final avg score: {checkpoint.get('final_avg_score', 'unknown')}")
    print(f"  Input size: {checkpoint.get('input_size', 'unknown')}")
    print(f"  File size: {model_path.stat().st_size / 1024:.1f} KB")
else:
    print("Model not found")

