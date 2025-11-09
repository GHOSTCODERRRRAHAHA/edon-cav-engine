#!/usr/bin/env python3
"""Test the Adaptive Memory Engine through the API."""

import requests
import json
import pandas as pd

API_URL = "http://localhost:8000"

def test_memory_api():
    """Test the memory engine through the API."""
    print("=" * 60)
    print("Testing Adaptive Memory Engine via API")
    print("=" * 60)
    
    # Check root endpoint
    print("\n1. Checking root endpoint...")
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        data = response.json()
        print(f"   Service: {data.get('service')}")
        print(f"   Version: {data.get('version')}")
        endpoints = data.get('endpoints', {})
        print(f"   Endpoints available:")
        for key, value in endpoints.items():
            print(f"      {key}: {value}")
        
        # Check if memory endpoints are present
        if 'memory_summary' in endpoints and 'memory_clear' in endpoints:
            print("   [OK] Memory endpoints are available!")
        else:
            print("   [WARN] Memory endpoints not found")
    except Exception as e:
        print(f"   [ERROR] {e}")
        return
    
    # Clear memory first
    print("\n2. Clearing memory...")
    try:
        response = requests.post(f"{API_URL}/memory/clear", timeout=5)
        if response.status_code == 200:
            print("   [OK] Memory cleared")
        else:
            print(f"   [WARN] Status {response.status_code}")
    except Exception as e:
        print(f"   [WARN] Could not clear memory: {e}")
    
    # Load a test window from dataset
    print("\n3. Loading test window from dataset...")
    try:
        with open("outputs/oem_sample_windows.jsonl", "r") as f:
            first_window = json.loads(f.readline())
        print(f"   Window ID: {first_window['window_id']}")
    except Exception as e:
        print(f"   [ERROR] Could not load dataset: {e}")
        return
    
    # Make CAV requests to populate memory
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
                print(f"      Adaptive: z={adaptive['z_cav']:.2f}, sens={adaptive['sensitivity']:.2f}, env_adj={adaptive['env_weight_adj']:.2f}")
            else:
                print(f"      [WARN] No adaptive field in response")
                print(f"      CAV: {result['cav_smooth']}, State: {result['state']}")
            
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
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Memory engine API test complete!")
    print("=" * 60)
    
    if adaptive_results:
        print(f"\nAdaptive results summary:")
        print(f"   Windows processed: {len(adaptive_results)}")
        avg_z = sum(r['z_cav'] for r in adaptive_results) / len(adaptive_results)
        print(f"   Average z-score: {avg_z:.2f}")

if __name__ == "__main__":
    test_memory_api()

