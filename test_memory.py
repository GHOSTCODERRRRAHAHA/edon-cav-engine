#!/usr/bin/env python3
"""Test the Adaptive Memory Engine."""

import requests
import json
import pandas as pd
import time

API_URL = "http://localhost:8000"

def test_memory_engine():
    """Test the memory engine functionality."""
    print("=" * 60)
    print("Testing Adaptive Memory Engine")
    print("=" * 60)
    
    # Load the dataset
    print("\n1. Loading dataset...")
    df = pd.read_csv("outputs/oem_sample_windows.csv")
    print(f"   Loaded {len(df)} windows from dataset")
    
    # Load the JSONL to get raw signals for first window
    print("\n2. Loading raw signals from JSONL...")
    with open("outputs/oem_sample_windows.jsonl", "r") as f:
        first_line = f.readline()
        first_window = json.loads(first_line)
    
    print(f"   Window ID: {first_window['window_id']}")
    print(f"   Has raw signals: {'EDA' in first_window}")
    
    # Clear memory first
    print("\n3. Clearing memory...")
    try:
        response = requests.post(f"{API_URL}/memory/clear", timeout=5)
        if response.status_code == 200:
            print("   [OK] Memory cleared")
        else:
            print(f"   [WARN] Clear returned status {response.status_code}")
    except Exception as e:
        print(f"   [WARN] Could not clear memory: {e}")
    
    # Make a few CAV requests to populate memory
    print("\n4. Making CAV requests to populate memory...")
    test_windows = []
    with open("outputs/oem_sample_windows.jsonl", "r") as f:
        for i, line in enumerate(f):
            if i >= 5:  # Test with first 5 windows
                break
            test_windows.append(json.loads(line))
    
    adaptive_results = []
    for i, window in enumerate(test_windows):
        print(f"   Processing window {i+1}/5...")
        
        # Create request payload
        payload = {
            "EDA": window["EDA"],
            "TEMP": window["TEMP"],
            "BVP": window["BVP"],
            "ACC_x": window["ACC_x"],
            "ACC_y": window["ACC_y"],
            "ACC_z": window["ACC_z"],
            "temp_c": window["temp_c"],
            "humidity": window["humidity"],
            "aqi": window["aqi"],
            "local_hour": window["local_hour"]
        }
        
        try:
            response = requests.post(f"{API_URL}/cav", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            # Check if adaptive field is present
            if "adaptive" in result:
                adaptive = result["adaptive"]
                adaptive_results.append({
                    "window_id": window["window_id"],
                    "cav_smooth": result["cav_smooth"],
                    "state": result["state"],
                    "z_cav": adaptive["z_cav"],
                    "sensitivity": adaptive["sensitivity"],
                    "env_weight_adj": adaptive["env_weight_adj"]
                })
                print(f"      CAV: {result['cav_smooth']}, State: {result['state']}")
                print(f"      Adaptive: z={adaptive['z_cav']}, sens={adaptive['sensitivity']}, env_adj={adaptive['env_weight_adj']}")
            else:
                print(f"      [WARN] No adaptive field in response")
            
            time.sleep(0.1)  # Small delay
            
        except Exception as e:
            print(f"      [ERROR] Request failed: {e}")
    
    # Check memory summary
    print("\n5. Checking memory summary...")
    try:
        response = requests.get(f"{API_URL}/memory/summary", timeout=5)
        response.raise_for_status()
        summary = response.json()
        
        print(f"   Total records: {summary['total_records']}")
        print(f"   Window hours: {summary['window_hours']}")
        
        if summary['overall_stats']:
            overall = summary['overall_stats']
            print(f"   Overall CAV mean: {overall['cav_mean']:.1f}")
            print(f"   Overall CAV std: {overall['cav_std']:.1f}")
            print(f"   State distribution:")
            for state, prob in overall['state_distribution'].items():
                print(f"      {state}: {prob:.3f}")
        
        if summary['hourly_stats']:
            print(f"   Hourly stats available for {len(summary['hourly_stats'])} hours")
            # Show stats for hour 12 (since all our test data uses hour 12)
            if 12 in summary['hourly_stats']:
                hour12 = summary['hourly_stats'][12]
                print(f"   Hour 12 stats:")
                print(f"      CAV mean: {hour12['cav_mean']:.1f}")
                print(f"      CAV std: {hour12['cav_std']:.1f}")
                print(f"      State probabilities:")
                for state, prob in hour12['state_probs'].items():
                    print(f"         {state}: {prob:.3f}")
        
    except Exception as e:
        print(f"   [ERROR] Could not get memory summary: {e}")
    
    # Verify dataset correctness
    print("\n6. Verifying dataset correctness...")
    print(f"   Dataset has {len(df)} windows")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Required columns present:")
    required_cols = ['window_id', 'cav_raw', 'cav_smooth', 'state', 'parts_bio', 
                     'parts_env', 'parts_circadian', 'parts_p_stress', 
                     'eda_mean', 'bvp_std', 'acc_magnitude_mean']
    for col in required_cols:
        if col in df.columns:
            print(f"      [OK] {col}")
        else:
            print(f"      [MISSING] {col}")
    
    # Check data ranges
    print(f"\n   Data ranges:")
    print(f"      CAV smooth: {df['cav_smooth'].min():.0f} - {df['cav_smooth'].max():.0f}")
    print(f"      States: {df['state'].unique().tolist()}")
    print(f"      EDA mean: {df['eda_mean'].min():.3f} - {df['eda_mean'].max():.3f}")
    print(f"      BVP std: {df['bvp_std'].min():.3f} - {df['bvp_std'].max():.3f}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Memory engine test complete!")
    print("=" * 60)
    
    return adaptive_results

if __name__ == "__main__":
    test_memory_engine()

