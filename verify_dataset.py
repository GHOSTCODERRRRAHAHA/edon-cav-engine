#!/usr/bin/env python3
"""Verify the 10-window dataset."""

import pandas as pd
import json

print("=" * 60)
print("Dataset Verification")
print("=" * 60)

# Verify CSV
print("\n1. CSV Dataset:")
df = pd.read_csv("outputs/oem_sample_windows.csv")
print(f"   Windows: {len(df)}")
print(f"   Columns: {len(df.columns)}")
print(f"   CAV range: {df['cav_smooth'].min():.0f} - {df['cav_smooth'].max():.0f}")
print(f"   States: {df['state'].unique().tolist()}")

required_cols = ['window_id', 'cav_raw', 'cav_smooth', 'state', 'parts_bio', 
                 'parts_env', 'parts_circadian', 'parts_p_stress', 
                 'eda_mean', 'bvp_std', 'acc_magnitude_mean']
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    print(f"   [ERROR] Missing columns: {missing_cols}")
else:
    print(f"   [OK] All required columns present")

# Verify JSONL
print("\n2. JSONL Dataset:")
with open("outputs/oem_sample_windows.jsonl", "r") as f:
    lines = f.readlines()
    windows = [json.loads(line) for line in lines]

print(f"   Lines: {len(windows)}")
if windows:
    first = windows[0]
    print(f"   First window ID: {first.get('window_id', 'N/A')}")
    print(f"   Has raw signals: {'EDA' in first}")
    print(f"   Has CAV fields: {'cav_smooth' in first}")
    print(f"   Has adaptive fields: {'parts_bio' in first}")
    print(f"   Signal array lengths:")
    for sig in ['EDA', 'TEMP', 'BVP', 'ACC_x', 'ACC_y', 'ACC_z']:
        if sig in first:
            print(f"      {sig}: {len(first[sig])} samples")

# Verify Parquet
print("\n3. Parquet Dataset:")
try:
    df_parquet = pd.read_parquet("outputs/oem_sample_windows.parquet")
    print(f"   Windows: {len(df_parquet)}")
    print(f"   Columns: {len(df_parquet.columns)}")
    print(f"   [OK] Parquet file readable")
except Exception as e:
    print(f"   [ERROR] Could not read Parquet: {e}")

print("\n" + "=" * 60)
print("[SUCCESS] Dataset verification complete!")
print("=" * 60)

