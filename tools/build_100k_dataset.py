#!/usr/bin/env python3
"""
Resumable 100K Dataset Builder

Builds a 100,000 window OEM dataset using batch API calls.
Supports resuming from checkpoint if interrupted.
"""

import os
import sys
import json
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import time

# Configuration
WINDOW_SIZE = 240  # 240 samples (60 seconds at 4 Hz)
BATCH_SIZE = 100  # Number of windows per batch request
API_URL = "http://localhost:8000/oem/cav/batch"
INPUT_CSV_PATHS = [
    "data/real_wesad.csv",
    "sensors/real_wesad.csv",
]
OUTPUT_DIR = Path("outputs")
OUTPUT_CSV = OUTPUT_DIR / "oem_100k_windows.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "oem_100k_windows.parquet"
CHECKPOINT_FILE = OUTPUT_DIR / "build_100k_checkpoint.json"

# Default environmental parameters
DEFAULT_TEMP_C = 24.0
DEFAULT_HUMIDITY = 50.0
DEFAULT_AQI = 42
DEFAULT_LOCAL_HOUR = 12

# Target number of windows
TARGET_WINDOWS = 100000


def find_input_file() -> str:
    """Find the input CSV file."""
    for path in INPUT_CSV_PATHS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Input file not found. Checked: {', '.join(INPUT_CSV_PATHS)}")


def load_checkpoint() -> Dict[str, Any]:
    """Load checkpoint if exists."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'processed_windows': 0, 'records': []}


def save_checkpoint(processed_windows: int, records: List[Dict[str, Any]]):
    """Save checkpoint."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'processed_windows': processed_windows,
        'total_records': len(records),
        'timestamp': time.time()
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def load_sensor_data(csv_path: str) -> pd.DataFrame:
    """Load and validate sensor data from CSV."""
    print(f"Loading sensor data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Convert all columns to float
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Validate required columns
    required_cols = ["EDA", "TEMP", "BVP", "ACC_x", "ACC_y", "ACC_z"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
    return df


def compute_acc_magnitude(acc_x: List[float], acc_y: List[float], acc_z: List[float]) -> List[float]:
    """Compute acceleration magnitude."""
    return [np.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(acc_x, acc_y, acc_z)]


def compute_analytics(window: Dict[str, List[float]]) -> Dict[str, float]:
    """Compute analytics for a sensor window."""
    eda_mean = np.mean(window["EDA"])
    bvp_std = np.std(window["BVP"])
    acc_magnitude = compute_acc_magnitude(
        window["ACC_x"], window["ACC_y"], window["ACC_z"]
    )
    acc_magnitude_mean = np.mean(acc_magnitude)
    
    return {
        "eda_mean": float(eda_mean),
        "bvp_std": float(bvp_std),
        "acc_magnitude_mean": float(acc_magnitude_mean),
    }


def create_window_payload(
    window: Dict[str, List[float]],
    temp_c: float = DEFAULT_TEMP_C,
    humidity: float = DEFAULT_HUMIDITY,
    aqi: int = DEFAULT_AQI,
    local_hour: int = DEFAULT_LOCAL_HOUR,
) -> Dict[str, Any]:
    """Create API request payload for a sensor window."""
    return {
        "EDA": window["EDA"],
        "TEMP": window["TEMP"],
        "BVP": window["BVP"],
        "ACC_x": window["ACC_x"],
        "ACC_y": window["ACC_y"],
        "ACC_z": window["ACC_z"],
        "temp_c": temp_c,
        "humidity": humidity,
        "aqi": aqi,
        "local_hour": local_hour,
    }


def call_batch_api(windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Call the batch API endpoint."""
    try:
        payload = {"windows": windows}
        response = requests.post(API_URL, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        return result["results"]
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Batch API call failed: {e}")


def extract_window_data(
    window_id: int,
    window_start_idx: int,
    window: Dict[str, List[float]],
    api_response: Dict[str, Any],
    analytics: Dict[str, float],
    temp_c: float,
    humidity: float,
    aqi: int,
    local_hour: int,
) -> Dict[str, Any]:
    """Extract all data fields for a single window."""
    parts = api_response.get("parts", {})
    
    return {
        "window_id": window_id,
        "window_start_idx": window_start_idx,
        "cav_raw": api_response.get("cav_raw"),
        "cav_smooth": api_response.get("cav_smooth"),
        "state": api_response.get("state"),
        "parts_bio": parts.get("bio", None),
        "parts_env": parts.get("env", None),
        "parts_circadian": parts.get("circadian", None),
        "parts_p_stress": parts.get("p_stress", None),
        "temp_c": temp_c,
        "humidity": humidity,
        "aqi": aqi,
        "local_hour": local_hour,
        "eda_mean": analytics["eda_mean"],
        "bvp_std": analytics["bvp_std"],
        "acc_magnitude_mean": analytics["acc_magnitude_mean"],
    }


def build_dataset_resumable(
    df: pd.DataFrame,
    start_from: int = 0,
    target_windows: int = TARGET_WINDOWS,
    temp_c: float = DEFAULT_TEMP_C,
    humidity: float = DEFAULT_HUMIDITY,
    aqi: int = DEFAULT_AQI,
    local_hour: int = DEFAULT_LOCAL_HOUR,
    batch_size: int = BATCH_SIZE,
    checkpoint_interval: int = 1000,
) -> pd.DataFrame:
    """
    Build dataset with resumable checkpoint support.
    
    Args:
        df: Sensor data DataFrame
        start_from: Starting window index (for resuming)
        target_windows: Target number of windows
        temp_c: Environmental temperature
        humidity: Humidity percentage
        aqi: Air Quality Index
        local_hour: Local hour [0-23]
        batch_size: Number of windows per batch request
        checkpoint_interval: Save checkpoint every N windows
    
    Returns:
        DataFrame with all processed windows
    """
    max_windows = len(df) - WINDOW_SIZE + 1
    target_windows = min(target_windows, max_windows)
    
    records: List[Dict[str, Any]] = []
    processed = start_from
    
    print(f"Building dataset: {target_windows:,} windows")
    print(f"Starting from window: {start_from:,}")
    print(f"Batch size: {batch_size} windows per request")
    print(f"Checkpoint interval: {checkpoint_interval:,} windows")
    print()
    
    # Use sliding window with stride to maximize data usage
    # For 100k windows, we'll use a stride of 1 (every window)
    stride = max(1, max_windows // target_windows)
    
    with tqdm(total=target_windows, initial=processed, desc="Processing windows") as pbar:
        while processed < target_windows:
            # Calculate window indices
            window_start_idx = (processed * stride) % max_windows
            window_end_idx = window_start_idx + WINDOW_SIZE
            
            # Check if we have enough data
            if window_end_idx > len(df):
                # Wrap around or stop
                if processed > 0:
                    print(f"\n[WARNING] Reached end of data at window {processed:,}")
                    print(f"  Processed {processed:,} windows (target: {target_windows:,})")
                    break
                else:
                    raise ValueError(f"Not enough data: need {WINDOW_SIZE} rows, have {len(df)}")
            
            # Prepare batch
            batch_windows = []
            batch_window_data = []
            batch_size_actual = min(batch_size, target_windows - processed)
            
            for i in range(batch_size_actual):
                if processed + i >= target_windows:
                    break
                
                idx = ((processed + i) * stride) % max_windows
                if idx + WINDOW_SIZE > len(df):
                    break
                
                window = {
                    "EDA": df["EDA"].iloc[idx:idx + WINDOW_SIZE].tolist(),
                    "TEMP": df["TEMP"].iloc[idx:idx + WINDOW_SIZE].tolist(),
                    "BVP": df["BVP"].iloc[idx:idx + WINDOW_SIZE].tolist(),
                    "ACC_x": df["ACC_x"].iloc[idx:idx + WINDOW_SIZE].tolist(),
                    "ACC_y": df["ACC_y"].iloc[idx:idx + WINDOW_SIZE].tolist(),
                    "ACC_z": df["ACC_z"].iloc[idx:idx + WINDOW_SIZE].tolist(),
                }
                
                payload = create_window_payload(
                    window, temp_c=temp_c, humidity=humidity, aqi=aqi, local_hour=local_hour
                )
                batch_windows.append(payload)
                batch_window_data.append((processed + i, idx, window))
            
            if not batch_windows:
                break
            
            # Call batch API
            try:
                batch_results = call_batch_api(batch_windows)
            except Exception as e:
                print(f"\n[ERROR] Batch API call failed: {e}")
                print(f"  Saving checkpoint at {processed:,} windows...")
                save_checkpoint(processed, records)
                raise
            
            # Process results
            for (window_id, window_start_idx, window), api_response in zip(batch_window_data, batch_results):
                analytics = compute_analytics(window)
                record = extract_window_data(
                    window_id, window_start_idx, window, api_response,
                    analytics, temp_c, humidity, aqi, local_hour
                )
                records.append(record)
            
            processed += len(batch_windows)
            pbar.update(len(batch_windows))
            
            # Save checkpoint periodically
            if processed % checkpoint_interval == 0:
                save_checkpoint(processed, records)
                pbar.set_postfix({'checkpoint': 'saved'})
    
    # Final checkpoint
    save_checkpoint(processed, records)
    
    # Convert to DataFrame
    dataset_df = pd.DataFrame(records)
    return dataset_df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Resumable 100K OEM Dataset Builder"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=TARGET_WINDOWS,
        help=f"Target number of windows (default: {TARGET_WINDOWS:,})"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Number of windows per batch request (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Save checkpoint every N windows (default: 1000)"
    )
    parser.add_argument(
        "--clear-checkpoint",
        action="store_true",
        help="Clear existing checkpoint and start fresh"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Resumable 100K Dataset Builder")
    print("=" * 60)
    print()
    
    # Check API availability
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        health_response.raise_for_status()
        print("[OK] CAV API is available")
    except requests.exceptions.RequestException:
        print("[ERROR] CAV API is not available at http://localhost:8000")
        print("   Please start the FastAPI server first:")
        print("   .\\run_api.ps1")
        sys.exit(1)
    
    # Handle checkpoint
    start_from = 0
    if args.clear_checkpoint and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("[INFO] Cleared existing checkpoint")
    elif args.resume and CHECKPOINT_FILE.exists():
        checkpoint = load_checkpoint()
        start_from = checkpoint['processed_windows']
        print(f"[INFO] Resuming from checkpoint: {start_from:,} windows processed")
    
    # Load sensor data
    input_path = find_input_file()
    df = load_sensor_data(input_path)
    
    # Check if we have enough data
    if len(df) < WINDOW_SIZE:
        raise ValueError(f"Not enough data: {len(df)} rows, need at least {WINDOW_SIZE}")
    
    # Build dataset
    dataset_df = build_dataset_resumable(
        df,
        start_from=start_from,
        target_windows=args.target,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval
    )
    
    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving dataset...")
    print(f"  CSV: {OUTPUT_CSV}")
    dataset_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"  Parquet: {OUTPUT_PARQUET}")
    dataset_df.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
    
    # Remove checkpoint file on success
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print(f"  Checkpoint cleared")
    
    # Print summary
    print("\n" + "=" * 60)
    print("[SUCCESS] Built OEM dataset:")
    print(f"   Windows processed: {len(dataset_df):,}")
    print(f"   Columns: {len(dataset_df.columns)}")
    print(f"   Saved CSV and Parquet formats")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {OUTPUT_CSV}")
    print(f"  - {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()

