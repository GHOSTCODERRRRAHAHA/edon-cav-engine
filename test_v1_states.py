"""Test script for v1 state classifier - test all states."""

import requests
import json

# Server URL (v1 endpoint)
url = "http://127.0.0.1:8000/oem/cav/batch"

# Test window 1: Restorative (p_stress < 0.2)
# Very low EDA, stable signals, comfortable environment
window_restorative = {
    "EDA": [0.01 * i for i in range(240)],  # Very low EDA
    "TEMP": [36.5] * 240,  # Normal body temp
    "BVP": [0.5 + 0.1 * (i % 20) / 20 for i in range(240)],  # Stable BVP
    "ACC_x": [0.0] * 240,
    "ACC_y": [0.0] * 240,
    "ACC_z": [1.0] * 240,  # Minimal motion
    "temp_c": 22.0,  # Comfortable
    "humidity": 45.0
}

# Test window 2: Focus (0.2 <= p_stress <= 0.5 AND env >= 0.8 AND circadian >= 0.9)
# Moderate EDA, good environment, optimal conditions
window_focus = {
    "EDA": [0.03 * i for i in range(240)],  # Moderate EDA
    "TEMP": [36.8] * 240,  # Slightly elevated
    "BVP": [0.5 + 0.2 * (i % 15) / 15 for i in range(240)],  # Rhythmic BVP
    "ACC_x": [0.05] * 240,  # Small movements
    "ACC_y": [0.02] * 240,
    "ACC_z": [1.0] * 240,
    "temp_c": 20.0,  # Cool, optimal
    "humidity": 40.0  # Low humidity, good
}

# Test window 3: Balanced (p_stress < 0.8, but not focus conditions)
# Moderate stress without strong alignment
window_balanced = {
    "EDA": [0.05 * i for i in range(240)],  # Moderate-high EDA
    "TEMP": [37.0] * 240,  # Elevated
    "BVP": [0.5 + 0.3 * (i % 12) / 12 for i in range(240)],  # Variable BVP
    "ACC_x": [0.1] * 240,
    "ACC_y": [0.05] * 240,
    "ACC_z": [1.1] * 240,
    "temp_c": 25.0,  # Moderate temp
    "humidity": 60.0  # Moderate humidity
}

# Test window 4: Overload (p_stress >= 0.8)
# Very high EDA, extreme signals, harsh environment
window_overload = {
    "EDA": [0.15 * i for i in range(240)],  # Very high EDA
    "TEMP": [37.5] * 240,  # High body temp
    "BVP": [0.5 + 0.5 * (i % 8) / 8 for i in range(240)],  # Erratic BVP
    "ACC_x": [0.5] * 240,  # High motion
    "ACC_y": [0.3] * 240,
    "ACC_z": [1.5] * 240,
    "temp_c": 35.0,  # Very hot
    "humidity": 90.0  # Very humid
}

def test_window(name, window):
    """Test a single window and print results."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    try:
        response = requests.post(url, json={"windows": [window]}, timeout=5)
        response.raise_for_status()
        result = response.json()
        
        if "results" in result and len(result["results"]) > 0:
            item = result["results"][0]
            if item.get("ok", False):
                print(f"✓ Request successful")
                print(f"State: {item.get('state', 'N/A')}")
                print(f"CAV Raw: {item.get('cav_raw', 'N/A')}")
                print(f"CAV Smooth: {item.get('cav_smooth', 'N/A')}")
                if "parts" in item:
                    parts = item["parts"]
                    print(f"Parts:")
                    print(f"  p_stress: {parts.get('p_stress', 'N/A')}")
                    print(f"  bio: {parts.get('bio', 'N/A')}")
                    print(f"  env: {parts.get('env', 'N/A')}")
                    print(f"  circadian: {parts.get('circadian', 'N/A')}")
            else:
                print(f"✗ Request failed: {item.get('error', 'Unknown error')}")
        else:
            print("Unexpected response format:")
            print(json.dumps(result, indent=2))
            
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to {url}")
        print("Make sure the server is running with:")
        print("  python -m uvicorn app.main:app --port 8000")
        print("  (or set EDON_MODE=v1)")
    except Exception as e:
        print(f"ERROR: {e}")
        if 'response' in locals():
            print(f"Response: {response.text}")

if __name__ == "__main__":
    print("EDON v1 State Classifier Test")
    print("=" * 60)
    print("Testing all v1 states: restorative, focus, balanced, overload")
    print("=" * 60)
    
    # Test all states
    test_window("Restorative State (p_stress < 0.2)", window_restorative)
    test_window("Focus State (0.2 <= p_stress <= 0.5, env>=0.8, circadian>=0.9)", window_focus)
    test_window("Balanced State (p_stress < 0.8, not focus)", window_balanced)
    test_window("Overload State (p_stress >= 0.8)", window_overload)
    
    print(f"\n{'='*60}")
    print("Test Complete")
    print(f"{'='*60}")
    print("\nExpected states:")
    print("  - Restorative: p_stress < 0.2")
    print("  - Focus: 0.2 <= p_stress <= 0.5 AND env >= 0.8 AND circadian >= 0.9")
    print("  - Balanced: p_stress < 0.8 (and not focus)")
    print("  - Overload: p_stress >= 0.8")

