#!/usr/bin/env python3
"""Quick analysis of the OEM dataset."""

import pandas as pd
import json

# Load CSV
df = pd.read_csv('outputs/oem_sample_windows.csv')

print("=" * 60)
print("OEM Dataset Analysis")
print("=" * 60)
print(f"\nTotal Records: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"\nColumn Names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

print("\n" + "=" * 60)
print("Dataset Contents:")
print("=" * 60)

print("\n1. WINDOW METADATA:")
print(f"   - window_id: Unique identifier for each window")
print(f"   - window_start_idx: Starting index in source data")

print("\n2. CAV OUTPUTS:")
print(f"   - cav_raw: Raw CAV score [0-10000]")
print(f"     Range: {df['cav_raw'].min()} - {df['cav_raw'].max()}")
print(f"   - cav_smooth: EMA-smoothed CAV score [0-10000]")
print(f"     Range: {df['cav_smooth'].min()} - {df['cav_smooth'].max()}")
print(f"   - state: User state classification")
print(f"     Distribution:")
for state, count in df['state'].value_counts().items():
    pct = (count / len(df)) * 100
    print(f"       {state:12s}: {count:5,} ({pct:5.1f}%)")

print("\n3. COMPONENT PARTS:")
print(f"   - parts_bio: Biological component [0-1]")
print(f"     Range: {df['parts_bio'].min():.3f} - {df['parts_bio'].max():.3f}")
print(f"   - parts_env: Environmental component [0-1]")
print(f"     Range: {df['parts_env'].min():.3f} - {df['parts_env'].max():.3f}")
print(f"   - parts_circadian: Circadian component [0-1]")
print(f"     Range: {df['parts_circadian'].min():.3f} - {df['parts_circadian'].max():.3f}")
print(f"   - parts_p_stress: Probability of stress [0-1]")
print(f"     Range: {df['parts_p_stress'].min():.6f} - {df['parts_p_stress'].max():.6f}")

print("\n4. ENVIRONMENTAL CONTEXT:")
print(f"   - temp_c: Temperature in Celsius")
print(f"     Range: {df['temp_c'].min():.1f} - {df['temp_c'].max():.1f}°C")
print(f"   - humidity: Humidity percentage")
print(f"     Range: {df['humidity'].min():.1f} - {df['humidity'].max():.1f}%")
print(f"   - aqi: Air Quality Index")
print(f"     Range: {df['aqi'].min()} - {df['aqi'].max()}")
print(f"   - local_hour: Local hour of day [0-23]")
print(f"     Range: {int(df['local_hour'].min())} - {int(df['local_hour'].max())}")

print("\n5. ANALYTICS (computed from raw signals):")
print(f"   - eda_mean: Mean EDA value")
print(f"     Range: {df['eda_mean'].min():.3f} - {df['eda_mean'].max():.3f}")
print(f"   - bvp_std: Standard deviation of BVP")
print(f"     Range: {df['bvp_std'].min():.3f} - {df['bvp_std'].max():.3f}")
print(f"   - acc_magnitude_mean: Mean acceleration magnitude")
print(f"     Range: {df['acc_magnitude_mean'].min():.3f} - {df['acc_magnitude_mean'].max():.3f}")

# Check JSONL for raw signals
print("\n" + "=" * 60)
print("JSONL Format (includes raw signals):")
print("=" * 60)
with open('outputs/oem_sample_windows.jsonl', 'r') as f:
    first_line = f.readline()
    data = json.loads(first_line)
    
print(f"\nEach JSONL record contains all CSV columns PLUS:")
print(f"   - EDA: {len(data['EDA'])} samples (240 samples = 60 seconds @ 4 Hz)")
print(f"   - TEMP: {len(data['TEMP'])} samples")
print(f"   - BVP: {len(data['BVP'])} samples")
print(f"   - ACC_x: {len(data['ACC_x'])} samples")
print(f"   - ACC_y: {len(data['ACC_y'])} samples")
print(f"   - ACC_z: {len(data['ACC_z'])} samples")
print(f"\nTotal signal data per record: {len(data['EDA']) * 6:,} data points")

print("\n" + "=" * 60)
print("File Formats:")
print("=" * 60)
print("\n1. CSV (outputs/oem_sample_windows.csv):")
print("   - Analytics and results only (no raw signals)")
print("   - Easy to read, good for analysis")
print("   - Size: ~1.5 MB for 10,000 records")

print("\n2. Parquet (outputs/oem_sample_windows.parquet):")
print("   - Same as CSV but compressed binary format")
print("   - Faster to read/write, smaller file size")
print("   - Best for data processing pipelines")

print("\n3. JSONL (outputs/oem_sample_windows.jsonl):")
print("   - Full records WITH raw 240-sample signal arrays")
print("   - Complete data for model training/research")
print("   - Each line is a complete JSON record")
print("   - Size: ~50-100 MB for 10,000 records")

print("\n" + "=" * 60)

