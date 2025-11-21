"""Test script for v2 state classifier with different synthetic windows."""

import math
import random
import requests
import json

# Server URL
url = "http://127.0.0.1:8002/v2/oem/cav/batch"

# Test window 1: Restorative (very low stress, very low chaos)
# Low EDA, stable BVP, minimal motion, comfortable environment
window_rest = {
    "physio": {
        "EDA": [0.001 * i for i in range(240)],  # Very low EDA
        "BVP": [0.5 + 0.1 * math.sin(i / 20) for i in range(240)]  # Stable, low amplitude
    },
    "motion": {
        "ACC_x": [0.0] * 240,
        "ACC_y": [0.0] * 240,
        "ACC_z": [0.98] * 240  # Minimal motion, just gravity
    },
    "env": {
        "temp_c": 21.0,  # Comfortable
        "humidity": 40.0,  # Comfortable
        "aqi": 10  # Very clean air
    },
    "task": {
        "id": "rest",
        "difficulty": 0.0  # No task difficulty
    }
}

# Test window 2: Focus (moderate stress ~0.3-0.4, low chaos <0.2)
# Moderate EDA, rhythmic BVP, controlled motion, good environment
window_focus = {
    "physio": {
        "EDA": [0.03 * i for i in range(240)],  # Moderate EDA
        "BVP": [0.5 + 0.3 * math.sin(i / 8) for i in range(240)]  # Rhythmic, moderate amplitude
    },
    "motion": {
        "ACC_x": [0.05] * 240,  # Small controlled movements
        "ACC_y": [0.02] * 240,
        "ACC_z": [1.0] * 240
    },
    "env": {
        "temp_c": 20.0,  # Cool, good for focus
        "humidity": 35.0,  # Low humidity
        "aqi": 15  # Clean air
    },
    "task": {
        "id": "work",
        "difficulty": 0.5  # Moderate difficulty
    }
}

# Test window 3: Overload/Emergency (high stress >=0.8, high chaos >=0.6)
# Very high EDA, erratic BVP, chaotic motion, harsh environment
# Need p_stress >= 0.80 and p_chaos >= 0.60 for overload
# Need p_stress >= 0.90 and p_chaos >= 0.75 for emergency
# Note: EDA normalization caps at 2.0, so use values around 1.5-2.0 for high stress
window_overload = {
    "physio": {
        "EDA": [1.8 + 0.1 * random.uniform(-1, 1) for _ in range(240)],  # Very high EDA (near normalization cap)
        "BVP": [random.uniform(-1.5, 1.5) for _ in range(240)]  # Erratic BVP (high std)
    },
    "motion": {
        "ACC_x": [random.uniform(-8, 8) for _ in range(240)],  # Extreme chaotic motion (high std)
        "ACC_y": [random.uniform(-8, 8) for _ in range(240)],
        "ACC_z": [random.uniform(-6, 6) for _ in range(240)]
    },
    "env": {
        "temp_c": 38.0,  # Extreme heat
        "humidity": 95.0,  # Extreme humidity
        "aqi": 250  # Very poor air quality
    },
    "task": {
        "id": "emergency",
        "complexity": 1.0,  # Maximum complexity
        "deadline_proximity": 1.0  # Urgent deadline
    },
    "system": {
        "battery_level": 0.05,  # Critical battery
        "cpu_temp": 95.0,  # Critical CPU temp
        "system_load": 0.99  # Critical system load
    }
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
            print(f"State Class: {item.get('state_class', 'N/A')}")
            print(f"p_stress: {item.get('p_stress', 'N/A'):.3f}" if isinstance(item.get('p_stress'), (int, float)) else f"p_stress: {item.get('p_stress', 'N/A')}")
            print(f"p_chaos: {item.get('p_chaos', 'N/A'):.3f}" if isinstance(item.get('p_chaos'), (int, float)) else f"p_chaos: {item.get('p_chaos', 'N/A')}")
            print(f"p_focus: {item.get('p_focus', 'N/A'):.3f}" if isinstance(item.get('p_focus'), (int, float)) else f"p_focus: {item.get('p_focus', 'N/A')}")
            if "influences" in item:
                inf = item["influences"]
                print(f"Influences:")
                print(f"  speed_scale: {inf.get('speed_scale', 'N/A')}")
                print(f"  torque_scale: {inf.get('torque_scale', 'N/A')}")
                print(f"  caution_flag: {inf.get('caution_flag', 'N/A')}")
        else:
            print("Unexpected response format:")
            print(json.dumps(result, indent=2))
            
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to {url}")
        print("Make sure the server is running with:")
        print("  $env:EDON_MODE='v2'")
        print("  python -m uvicorn app.main:app --port 8002")
    except Exception as e:
        print(f"ERROR: {e}")
        print(f"Response: {response.text if 'response' in locals() else 'N/A'}")

if __name__ == "__main__":
    print("EDON v2 State Classifier Test")
    print("=" * 60)
    
    # Test all windows
    test_window("Restorative Window", window_rest)
    test_window("Focus Window", window_focus)
    test_window("Overload Window", window_overload)
    
    print(f"\n{'='*60}")
    print("Test Complete")
    print(f"{'='*60}")

