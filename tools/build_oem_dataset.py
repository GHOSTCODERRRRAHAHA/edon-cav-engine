#!/usr/bin/env python3
"""
EDON OEM Dataset Builder

Streams sensor windows from real_wesad.csv, calls the local /cav API,
and saves all results (state, cav, parts, environment) into clean dataset files.

Dataset Columns:
- Input signals: EDA, TEMP, BVP, ACC_x, ACC_y, ACC_z (240 samples each)
- CAV outputs: cav_raw, cav_smooth, state
- Component parts: parts_bio, parts_env, parts_circadian, parts_p_stress
- Environmental context: temp_c, humidity, aqi, local_hour
- Analytics: eda_mean, bvp_std, acc_magnitude_mean
- Window metadata: window_id, window_start_idx
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

# Configuration
WINDOW_SIZE = 240  # 240 samples (60 seconds at 4 Hz)
API_URL = "http://localhost:8000/cav"
INPUT_CSV_PATHS = [
    "data/real_wesad.csv",  # Preferred path per spec
    "sensors/real_wesad.csv",  # Fallback to existing location
]
OUTPUT_DIR = Path("outputs")
OUTPUT_CSV = OUTPUT_DIR / "oem_sample_windows.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "oem_sample_windows.parquet"
OUTPUT_JSONL = OUTPUT_DIR / "oem_sample_windows.jsonl"

# Default environmental parameters
DEFAULT_TEMP_C = 24.0
DEFAULT_HUMIDITY = 50.0
DEFAULT_AQI = 42
DEFAULT_LOCAL_HOUR = 12


def find_input_file() -> str:
    """Find the input CSV file, checking multiple possible locations."""
    for path in INPUT_CSV_PATHS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Input file not found. Checked: {', '.join(INPUT_CSV_PATHS)}"
    )


def load_sensor_data(csv_path: str) -> pd.DataFrame:
    """Load and validate sensor data from CSV."""
    print(f"Loading sensor data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Convert all columns to float (they're stored as strings)
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
    """Compute acceleration magnitude from x, y, z components."""
    return [np.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(acc_x, acc_y, acc_z)]


def compute_analytics(window: Dict[str, List[float]]) -> Dict[str, float]:
    """Compute light analytics for a sensor window."""
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


def call_cav_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call the local /cav API endpoint."""
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API call failed: {e}")


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
    include_raw_signals: bool = False,
) -> Dict[str, Any]:
    """Extract all data fields for a single window into a flat record."""
    parts = api_response.get("parts", {})
    
    record = {
        "window_id": window_id,
        "window_start_idx": window_start_idx,
        # CAV outputs
        "cav_raw": api_response.get("cav_raw"),
        "cav_smooth": api_response.get("cav_smooth"),
        "state": api_response.get("state"),
        # Component parts
        "parts_bio": parts.get("bio", None),
        "parts_env": parts.get("env", None),
        "parts_circadian": parts.get("circadian", None),
        "parts_p_stress": parts.get("p_stress", None),
        # Environmental context
        "temp_c": temp_c,
        "humidity": humidity,
        "aqi": aqi,
        "local_hour": local_hour,
        # Analytics
        "eda_mean": analytics["eda_mean"],
        "bvp_std": analytics["bvp_std"],
        "acc_magnitude_mean": analytics["acc_magnitude_mean"],
    }
    
    # Include raw signals for JSONL (for research/model training)
    if include_raw_signals:
        record.update({
            "EDA": window["EDA"],
            "TEMP": window["TEMP"],
            "BVP": window["BVP"],
            "ACC_x": window["ACC_x"],
            "ACC_y": window["ACC_y"],
            "ACC_z": window["ACC_z"],
        })
    
    return record


def build_dataset(
    df: pd.DataFrame,
    temp_c: float = DEFAULT_TEMP_C,
    humidity: float = DEFAULT_HUMIDITY,
    aqi: int = DEFAULT_AQI,
    local_hour: int = DEFAULT_LOCAL_HOUR,
    limit: int = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Build OEM dataset by processing all windows.
    
    Args:
        df: Sensor data DataFrame
        temp_c: Environmental temperature
        humidity: Humidity percentage
        aqi: Air Quality Index
        local_hour: Local hour [0-23]
        limit: Maximum number of windows to process (None for all)
    
    Returns:
        Tuple of (analytics DataFrame, full records with raw signals for JSONL)
    """
    num_windows = len(df) - WINDOW_SIZE + 1
    if limit is not None:
        num_windows = min(num_windows, limit)
    analytics_records: List[Dict[str, Any]] = []
    full_records: List[Dict[str, Any]] = []
    
    print(f"Processing {num_windows:,} windows...")
    
    for window_id in tqdm(range(num_windows), desc="Processing windows"):
        window_start_idx = window_id
        window_end_idx = window_start_idx + WINDOW_SIZE
        
        # Extract window data
        window = {
            "EDA": df["EDA"].iloc[window_start_idx:window_end_idx].tolist(),
            "TEMP": df["TEMP"].iloc[window_start_idx:window_end_idx].tolist(),
            "BVP": df["BVP"].iloc[window_start_idx:window_end_idx].tolist(),
            "ACC_x": df["ACC_x"].iloc[window_start_idx:window_end_idx].tolist(),
            "ACC_y": df["ACC_y"].iloc[window_start_idx:window_end_idx].tolist(),
            "ACC_z": df["ACC_z"].iloc[window_start_idx:window_end_idx].tolist(),
        }
        
        # Compute analytics
        analytics = compute_analytics(window)
        
        # Create API payload
        payload = create_window_payload(
            window, temp_c=temp_c, humidity=humidity, aqi=aqi, local_hour=local_hour
        )
        
        # Call API
        api_response = call_cav_api(payload)
        
        # Extract analytics record (for CSV/Parquet)
        analytics_record = extract_window_data(
            window_id,
            window_start_idx,
            window,
            api_response,
            analytics,
            temp_c,
            humidity,
            aqi,
            local_hour,
            include_raw_signals=False,
        )
        analytics_records.append(analytics_record)
        
        # Extract full record with raw signals (for JSONL)
        full_record = extract_window_data(
            window_id,
            window_start_idx,
            window,
            api_response,
            analytics,
            temp_c,
            humidity,
            aqi,
            local_hour,
            include_raw_signals=True,
        )
        full_records.append(full_record)
    
    # Convert to DataFrame
    dataset_df = pd.DataFrame(analytics_records)
    return dataset_df, full_records


def save_dataset(dataset_df: pd.DataFrame, all_windows_data: List[Dict[str, Any]]):
    """Save dataset to CSV, Parquet, and JSONL formats."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save CSV (analytics only, no raw signals)
    print(f"Saving CSV to: {OUTPUT_CSV}")
    dataset_df.to_csv(OUTPUT_CSV, index=False)
    
    # Save Parquet (analytics only, no raw signals)
    print(f"Saving Parquet to: {OUTPUT_PARQUET}")
    dataset_df.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
    
    # Save JSONL (with raw signals for research/model training)
    print(f"Saving JSONL to: {OUTPUT_JSONL}")
    with open(OUTPUT_JSONL, "w") as f:
        for record in all_windows_data:
            # Convert numpy/types to native Python types for JSON serialization
            json_record = {}
            for key, value in record.items():
                if value is None:
                    json_record[key] = None
                elif isinstance(value, (list, np.ndarray)):
                    # Handle lists and arrays (including signal arrays)
                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                    # Ensure list elements are native Python types
                    json_record[key] = [
                        float(v) if isinstance(v, (np.floating, np.float64, np.float32)) else
                        int(v) if isinstance(v, (np.integer, np.int64, np.int32)) else
                        None if (isinstance(v, float) and pd.isna(v)) else v
                        for v in value
                    ]
                elif isinstance(value, (np.integer, np.int64, np.int32)):
                    json_record[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    json_record[key] = float(value)
                elif isinstance(value, (int, float, str, bool)):
                    # Check for NaN only on scalar numeric values
                    if isinstance(value, float) and pd.isna(value):
                        json_record[key] = None
                    else:
                        json_record[key] = value
                else:
                    json_record[key] = value
            f.write(json.dumps(json_record) + "\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="EDON OEM Dataset Builder - Streams sensor windows and calls CAV API"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of windows to process (default: all windows)",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("EDON OEM Dataset Builder")
    print("=" * 60)
    if args.limit:
        print(f"Processing limit: {args.limit:,} windows")
    print()
    
    # Check API availability
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        health_response.raise_for_status()
        print("[OK] CAV API is available")
    except requests.exceptions.RequestException:
        print("[ERROR] CAV API is not available at http://localhost:8000")
        print("   Please start the FastAPI server first:")
        print("   uvicorn app.main:app --reload")
        sys.exit(1)
    
    # Load sensor data
    input_path = find_input_file()
    df = load_sensor_data(input_path)
    
    # Check if we have enough data
    if len(df) < WINDOW_SIZE:
        raise ValueError(f"Not enough data: {len(df)} rows, need at least {WINDOW_SIZE}")
    
    # Build dataset
    dataset_df, full_records = build_dataset(df, limit=args.limit)
    
    # Save outputs
    save_dataset(dataset_df, full_records)
    
    # Print summary
    print("\n" + "=" * 60)
    print("[SUCCESS] Built OEM dataset:")
    print(f"   Windows processed: {len(dataset_df):,}")
    print(f"   Columns: {len(dataset_df.columns)}")
    print(f"   Saved CSV, Parquet, and JSONL formats")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {OUTPUT_CSV}")
    print(f"  - {OUTPUT_PARQUET}")
    print(f"  - {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()

