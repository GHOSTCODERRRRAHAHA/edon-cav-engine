#!/usr/bin/env python3
"""Create OEM sample data files."""

import numpy as np
import pandas as pd
import json
from pathlib import Path

# Create data directory
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Generate 5 sample windows
np.random.seed(42)
windows = []

for i in range(5):
    window = {
        "EDA": np.random.normal(0, 0.2, 240).tolist(),
        "TEMP": np.random.normal(32, 0.2, 240).tolist(),
        "BVP": np.random.normal(0, 0.5, 240).tolist(),
        "ACC_x": np.random.normal(0, 0.05, 240).tolist(),
        "ACC_y": np.random.normal(0, 0.05, 240).tolist(),
        "ACC_z": np.random.normal(1, 0.05, 240).tolist(),
        "temp_c": float(22.0 + np.random.uniform(-2, 2)),
        "humidity": float(50.0 + np.random.uniform(-10, 10)),
        "aqi": int(35 + np.random.uniform(-10, 20)),
        "local_hour": int(np.random.randint(8, 20))
    }
    windows.append(window)

# Save as JSONL (one JSON object per line)
jsonl_path = data_dir / "oem_sample_windows.jsonl"
with open(jsonl_path, "w") as f:
    for window in windows:
        f.write(json.dumps(window) + "\n")
print(f"✓ Created {jsonl_path} with {len(windows)} windows")

# Save as Parquet (flatten arrays as strings for storage)
parquet_data = []
for window in windows:
    row = {
        "EDA": json.dumps(window["EDA"]),
        "TEMP": json.dumps(window["TEMP"]),
        "BVP": json.dumps(window["BVP"]),
        "ACC_x": json.dumps(window["ACC_x"]),
        "ACC_y": json.dumps(window["ACC_y"]),
        "ACC_z": json.dumps(window["ACC_z"]),
        "temp_c": window["temp_c"],
        "humidity": window["humidity"],
        "aqi": window["aqi"],
        "local_hour": window["local_hour"]
    }
    parquet_data.append(row)

df = pd.DataFrame(parquet_data)
parquet_path = data_dir / "oem_sample_windows.parquet"
df.to_parquet(parquet_path, index=False)
print(f"✓ Created {parquet_path} with {len(windows)} windows")

# Also create a single-window JSON for easy testing
single_window_path = data_dir / "oem_sample_window.json"
with open(single_window_path, "w") as f:
    json.dump(windows[0], f, indent=2)
print(f"✓ Created {single_window_path} (single window for testing)")

print(f"\n✓ Generated {len(windows)} sample windows")




